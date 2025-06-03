from dataclasses import dataclass
from typing import LiteralString, cast

import neo4j

import torch
from neo4j import GraphDatabase
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,  # pyright: ignore[reportPrivateImportUsage]
    AutoTokenizer,  # pyright: ignore[reportPrivateImportUsage]
)

import secret


@dataclass
class Tweet:
    node_id: str
    text: str
    tweet_id: str


@dataclass
class Labels:
    neg: float
    neu: float
    pos: float


def preprocess(text: str) -> str:
    new_text: list[str] = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


def get_tweets(driver: neo4j.Driver, skip: int, batch_size: int) -> list[Tweet]:
    # Fetch a batch of tweets
    batch_query = cast(
        LiteralString,
        f"""
            MATCH (t:Tweet)
            RETURN elementId(t) AS node_id, t.text AS text, t.id AS tweet_id
            SKIP {skip} LIMIT {batch_size}
        """,
    )

    records, _, _ = driver.execute_query(
        batch_query,
        database_="neo4j",
    )
    tweets: list[Tweet] = [
        Tweet(tweet["node_id"], tweet["text"], tweet["tweet_id"]) for tweet in records
    ]
    return tweets


def calculate_scores(model, tokenizer, text: str) -> tuple[Labels, float]:
    # Get the device that the model is on (CPU or GPU)
    device = next(model.parameters()).device

    # Tokenize and move to device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Keep computation on GPU
    scores = outputs.logits[0].softmax(dim=0)

    # Only move to CPU at the end
    scores_np = scores.cpu().numpy()

    labels = Labels(
        float(scores_np[0]),
        float(scores_np[1]),
        float(scores_np[2]),
    )
    sentiment_score = round(labels.pos - labels.neg, 4)
    return (labels, sentiment_score)


def update_sentiment(
    driver: neo4j.Driver,
    tweets: list[Tweet],
    labels: list[Labels],
    sentiment_scores: list[float],
) -> None:
    sentiment_labels: list[str] = []
    positives: list[float] = []
    neutrals: list[float] = []
    negatives: list[float] = []
    node_ids: list[str] = []
    ids: list[str] = []

    for label in labels:
        # Find the maximum value and set sentiment label directly
        value = {
            "positive": label.pos,
            "neutral": label.neu,
            "negative": label.neg,
        }
        sentiment_label = cast(str, max(value, key=value.get))  # pyright: ignore[reportArgumentType, reportCallIssue]
        sentiment_labels.append(sentiment_label)
        positives.append(label.pos)
        neutrals.append(label.neu)
        negatives.append(label.neg)

    for tweet in tweets:
        node_ids.append(tweet.node_id)
        ids.append(tweet.tweet_id)

    update_query: LiteralString = """
            UNWIND $node_ids as node_id, $positives as pos, $neutrals as neu, $negatives as neg, $scores as score, $labels as label $ids as id
            MATCH (t:Tweet) WHERE elementId(t) = node_ids
            SET t.positive = pos,
                t.neutral = neu,
                t.negative = neg,
                t.sentiment_score = score,
                t.sentiment_label = label,
                t.id = tweet_id
        """

    _ = driver.execute_query(
        update_query,
        node_ids=node_ids,
        positives=positives,  # Fixed variable name from label to labels
        neutrals=neutrals,
        negatives=negatives,
        score=sentiment_scores,
        label=sentiment_labels,
        ids=ids,
    )


def main() -> None:
    batch_size = 1000
    skip = 0
    labels: list[Labels] = []
    sentiment_scores: list[float] = []

    # ————— Load model & tokenizer —————
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    # Explicitly set use_fast=False and legacy=True to avoid tiktoken compatibility issues in Python 3.13
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    model = AutoModelForSequenceClassification.from_pretrained(model_name)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

    # ————— Neo4j HTTP setup —————
    with GraphDatabase.driver(secret.url, auth=("neo4j", secret.password)) as driver:
        driver.verify_connectivity()
        assert driver.verify_authentication()

        # while True:
        tweets = get_tweets(driver, skip, batch_size)
        # if not tweets:
        #     # No more tweets to process
        #     break

        # Show progress bar, no other prints
        for tweet in tqdm(
            tweets, desc=f"Processing batch {skip // batch_size + 1}", unit="tweet"
        ):
            text = preprocess(tweet.text)
            label, sentiment_score = calculate_scores(model, tokenizer, text)
            labels.append(label)
            sentiment_scores.append(sentiment_score)

            skip += batch_size
        update_sentiment(driver, tweets, labels, sentiment_scores)


if __name__ == "__main__":
    main()
