import sys
import os
from dotenv import load_dotenv

import neo4j

from transformers import XLMRobertaTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
import numpy as np
from tqdm import tqdm

# --- Configuration Constants ---
# Model
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# Neo4j
_ = load_dotenv()

NEO4J_BOLT_URL = os.getenv("NEO4J_BOLT_URL")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Batching
NEO4J_FETCH_BATCH_SIZE = 10000
INFERENCE_BATCH_SIZE = 64

# Progress Tracking
PROGRESS_FILE = "sentiment_progress.txt"

# --- Device Configuration ---
def configure_device() -> torch.device:
    """Configures and returns the appropriate torch device (CUDA if available, else CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

# --- Pre-Process function recommended by the authors ---
def preprocess(text: str) -> str:
    """
    Preprocesses the input text by replacing mentions (@user) and links (http)
    as recommended by the model authors.
    """
    if not isinstance(text, str):
        # Basic check for non-string types, return empty string or raise error
        print(f"Warning: Non-string input to preprocess: {type(text)}. Returning empty string.")
        return ""
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# --- Model, Tokenizer & Config Loading ---
def load_sentiment_model_and_tokenizer(model_name: str, device: torch.device):
    """
    Loads the XLM-RoBERTa tokenizer, AutoModelForSequenceClassification, and AutoConfig.

    model_name (str): The name of the pre-trained model.
    device (torch.device): The device (CPU/CUDA) to load the model onto.

    Returns:
        tuple: A tuple containing (tokenizer, model, config).
    """
    print(f"\nLoading model, tokenizer, and config for: {model_name}")
    try:
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_name, use_fast=True)
        print("Loaded fast XLMRobertaTokenizer.")
    except Exception as e:
        print(f"Failed to load fast XLMRobertaTokenizer: {e}")
        print("Falling back to slow tokenizer.")
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_name, use_fast=False)
        print("Loaded slow XLMRobertaTokenizer.")

    config = AutoConfig.from_pretrained(model_name) # To get label mapping (id2label)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device) # Move model to GPU/CPU
    model.eval()
    print("Model loaded successfully.")
    return tokenizer, model, config

# --- Neo4j Bolt Setup ---
def connect_to_neo4j(bolt_url: str, user: str, password: str) -> neo4j.GraphDatabase.driver:
    """
    Establishes and verifies a connection to the Neo4j Bolt database.

    bolt_url (str): The URL of the Neo4j Bolt server.
    user (str): The Neo4j username.
    password (str): The Neo4j password.

    Returns:
        neo4j.GraphDatabase.Driver: The Neo4j driver object.

    Raises:
        SystemExit: If connection fails.
    """
    print(f"\nAttempting to connect to Neo4j Bolt at {bolt_url}...")
    try:
        driver = neo4j.GraphDatabase.driver(bolt_url, auth=(user, password))
        driver.verify_connectivity()
        print("Successfully connected to Neo4j Bolt.")
        return driver
    except Exception as e:
        print(f"Error connecting to Neo4j Bolt at {bolt_url}: {e}")
        sys.exit("Exiting due to Neo4j connection error.")

# --- Resume Functionality & Progress File ---
def load_progress(filename: str) -> int:
    """Loads the last saved skip value from a file."""
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                return int(f.read().strip())
        except (ValueError, IOError) as e:
            print(f"Warning: Could not read progress file {filename}. Starting from 0. Error: {e}")
            return 0
    return 0 # Start from 0 if the file doesn't exist

def save_progress(filename: str, skip_value: int):
    """Saves the current skip value to a file."""
    try:
        with open(filename, "w") as f:
            f.write(str(skip_value))
    except IOError as e:
        print(f"Error: Could not save progress to file {filename}. Error: {e}")

# --- Neo4j Transaction Functions (passed to driver.session.execute_...) ---
def get_total_unprocessed_tweets_tx_fn(tx: neo4j.Transaction) -> int:
    """Transaction function to get the total count of tweets to process."""
    query = """
        MATCH (t:Tweet)
        WHERE t.sentiment_label IS NULL
        RETURN count(t) AS total_to_process
    """
    result = tx.run(query)
    record = result.single()
    return record["total_to_process"] if record else 0

def fetch_tweets_batch_tx_fn(tx: neo4j.Transaction, skip: int, limit: int) -> list[dict]:
    """Transaction function to fetch a batch of tweets."""
    query = f"""
        MATCH (t:Tweet)
        WHERE t.sentiment_label IS NULL
        RETURN id(t) AS node_id, t.text AS text, t.id AS tweet_id
        SKIP {skip} LIMIT {limit}
    """
    result = tx.run(query)
    return [record.data() for record in result]

def update_tweets_batch_tx_fn(tx: neo4j.Transaction, batch_data: list[dict]):
    """Transaction function to update a batch of tweets with sentiment data."""
    query = """
        UNWIND $batch_data AS params
        MATCH (t:Tweet) WHERE id(t) = params.nodeId
        SET t.positive = params.pos,
            t.neutral = params.neu,
            t.negative = params.negative,
            t.sentiment_score = params.sentiment_score,
            t.sentiment_label = params.sentiment_label
    """
    tx.run(query, batch_data=batch_data)

# --- Main Logic Functions ---
def get_total_unprocessed_tweets_count(driver: neo4j.GraphDatabase.driver) -> int:
    """Fetches the total count of tweets that still need sentiment analysis."""
    total_tweets_to_process = 0
    try:
        print("\nFetching total count of tweets to process via Bolt...")
        with driver.session() as session:
            total_tweets_to_process = session.execute_read(get_total_unprocessed_tweets_tx_fn)
        print(f"Total tweets matching criteria: {total_tweets_to_process}")
        return total_tweets_to_process
    except neo4j.exceptions.Neo4jError as e:
        print(f"Error fetching total count from Neo4j: {e}")
        sys.exit("Cannot show overall progress without total count. Exiting.")
    except Exception as e:
        print(f"An unexpected error occurred during total count fetch: {e}")
        sys.exit("Cannot show overall progress due to unexpected error. Exiting.")

def fetch_tweets_batch(driver: neo4j.GraphDatabase.driver, skip: int, limit: int) -> list[dict] | None:
    """Fetches a batch of tweets from Neo4j."""
    fetched_tweets_data = []
    try:
        print(f"\nFetching batch starting at skip {skip} (limit {limit}) via Bolt...")
        with driver.session() as session:
            fetched_tweets_data = session.execute_read(
                fetch_tweets_batch_tx_fn,
                skip,
                limit
            )
        return fetched_tweets_data
    except neo4j.exceptions.Neo4jError as e:
        print(f"Error fetching data from Neo4j for batch starting at {skip}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during Neo4j fetch: {e}")
        return None

def process_batch_for_sentiment(
    fetched_tweets_data: list[dict],
    tokenizer: XLMRobertaTokenizer,
    model: AutoModelForSequenceClassification,
    config: AutoConfig,
    device: torch.device,
    inference_batch_size: int
) -> list[dict]:
    """
    Processes a batch of fetched tweets for sentiment analysis.

    fetched_tweets_data (list[dict]): List of dictionaries, each containing 'node_id', 'text', 'tweet_id'.
    tokenizer, model, config: HuggingFace objects for sentiment analysis.
    device (torch.device): The device (CPU/CUDA) to run inference on.
    inference_batch_size (int): Batch size for model inference.

    Returns:
        list[dict]: A list of dictionaries, each containing sentiment data for a tweet,
                    ready for Neo4j update.
    """
    node_ids_batch = [tweet["node_id"] for tweet in fetched_tweets_data]
    tweet_ids_batch = [tweet["tweet_id"] for tweet in fetched_tweets_data]
    texts_batch = [preprocess(tweet["text"]) for tweet in fetched_tweets_data]

    all_update_params = []
    total_texts = len(texts_batch)

    # Process in smaller inference batches
    for i in tqdm(range(0, total_texts, inference_batch_size), desc=f"Inferring {total_texts} tweets"):
        sub_batch_texts = texts_batch[i : i + inference_batch_size]
        sub_batch_node_ids = node_ids_batch[i : i + inference_batch_size]
        sub_batch_tweet_ids = tweet_ids_batch[i : i + inference_batch_size]

        if not sub_batch_texts:
            continue

        # Tokenize batch
        inputs = tokenizer(
            sub_batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)

        # Perform model inference on batch
        with torch.no_grad():
            outputs = model(**inputs)

        # Get probabilities and calculate sentiment scores/labels
        probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy()

        for j in range(probabilities.shape[0]):
            scores = probabilities[j]
            neg, neu, pos = float(scores[0]), float(scores[1]), float(scores[2])

            sentiment_score_val = round(pos - neg, 4)
            predicted_class_id = np.argmax(scores)
            sentiment_label_val = config.id2label[predicted_class_id]

            all_update_params.append({
                "nodeId": sub_batch_node_ids[j],
                "pos": pos,
                "neu": neu,
                "negative": neg,
                "sentiment_score": sentiment_score_val,
                "sentiment_label": sentiment_label_val,
                "tweet_id": sub_batch_tweet_ids[j]
            })
    return all_update_params

def update_neo4j_with_sentiment(driver: neo4j.GraphDatabase.driver, update_params: list[dict], current_skip: int) -> bool:
    """
    Updates a batch of tweets in Neo4j with calculated sentiment data.

    driver (neo4j.GraphDatabase.Driver): The Neo4j driver object.
    update_params (list[dict]): List of dictionaries with sentiment data for updates.
    current_skip (int): The starting skip value for the current batch (for logging).

    Returns:
        bool: True if the update was successful, False otherwise.
    """
    if not update_params:
        print("No sentiment data to update for this batch.")
        return True # Consider it successful if nothing needed updating

    try:
        print(f"\nAttempting batch update of {len(update_params)} tweets via Bolt...")
        with driver.session() as session:
            session.execute_write(
                update_tweets_batch_tx_fn,
                batch_data=update_params
            )
        print(f"Successfully updated {len(update_params)} tweets in Neo4j for batch starting at {current_skip}.")
        return True
    except neo4j.exceptions.Neo4jError as e:
        print(f"Error updating Neo4j for batch starting at {current_skip}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during Neo4j update: {e}")
        return False

# --- Main Execution Function ---
def main():
    """
    Main function to orchestrate the sentiment analysis process for Neo4j tweets.
    """
    # 1. Device Configuration
    device = configure_device()

    # 2. Load Model, Tokenizer & Config
    tokenizer, model, config = load_sentiment_model_and_tokenizer(MODEL_NAME, device)

    # 3. Connect to Neo4j
    driver = connect_to_neo4j(NEO4J_BOLT_URL, NEO4J_USER, NEO4J_PASSWORD)

    # 4. Get Total Count of Tweets to Process
    total_tweets_to_process = get_total_unprocessed_tweets_count(driver)

    # 5. Load/Adjust Progress
    skip = load_progress(PROGRESS_FILE)
    print(f"Starting processing from skip: {skip}")

    # Adjust skip if the total count changed or points beyond current total
    if total_tweets_to_process > 0 and skip > total_tweets_to_process:
        print(f"Warning: Loaded skip value ({skip}) is greater than current total ({total_tweets_to_process}). Resetting skip to total.")
        skip = total_tweets_to_process
        save_progress(PROGRESS_FILE, skip)
    elif total_tweets_to_process == 0:
        print("No tweets found needing sentiment analysis. Exiting.")
        driver.close()
        return

    # 6. Main Processing Loop
    while True:
        # Fetch a batch of tweets from Neo4j
        fetched_tweets_data = fetch_tweets_batch(driver, skip, NEO4J_FETCH_BATCH_SIZE)

        if fetched_tweets_data is None: # An error occurred during fetch
            print("Stopping due to Neo4j fetch error.")
            break
        if not fetched_tweets_data:
            print(f"No more tweets matching criteria found in batch starting at skip {skip}. Assuming processing is complete.")
            break

        print(f"Fetched {len(fetched_tweets_data)} tweets from Neo4j (skip={skip}).")

        # Process the fetched batch for sentiment
        all_update_params = process_batch_for_sentiment(
            fetched_tweets_data, tokenizer, model, config, device, INFERENCE_BATCH_SIZE
        )

        # Update Neo4j with sentiment data
        update_successful = update_neo4j_with_sentiment(driver, all_update_params, skip)

        if update_successful:
            # Save progress AFTER successful update
            next_skip = skip + NEO4J_FETCH_BATCH_SIZE
            save_progress(PROGRESS_FILE, next_skip)

            # Report current progress
            processed_count = min(next_skip, total_tweets_to_process)
            print(f"Progress: {processed_count} / {total_tweets_to_process} tweets processed.")

            # Prepare for next iteration
            skip = next_skip
        else:
            print("Stopping due to Neo4j update error.")
            break # Break the main loop on update errors

    # 7. Close the Neo4j Driver
    if driver:
        print("\nClosing Neo4j driver connection.")
        driver.close()

    # Final message
    print("\nFinished processing all tweets.")
    final_processed = load_progress(PROGRESS_FILE)
    reported_processed = min(final_processed, total_tweets_to_process)
    print(f"Estimated processed based on last saved skip: {reported_processed} / {total_tweets_to_process}")


if __name__ == "__main__":
    main()