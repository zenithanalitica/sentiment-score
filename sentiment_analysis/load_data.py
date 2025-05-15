import requests
from requests.auth import HTTPBasicAuth
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from tqdm import tqdm
import secret

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# ————— Load model & tokenizer —————
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model     = AutoModelForSequenceClassification.from_pretrained(model_name)

# ————— Neo4j HTTP setup —————
auth = HTTPBasicAuth("neo4j", secret.password)

batch_size = 1000
skip = 0

while True:
    # Fetch a batch of tweets
    batch_query = {
        "statements": [
            {
                "statement": f"""
                    MATCH (t:Tweet)
                    RETURN id(t) AS node_id, t.text AS text, t.id AS tweet_id
                    SKIP {skip} LIMIT {batch_size}
                """
            }
        ]
    }

    response = requests.post(secret.url, json=batch_query, auth=auth)
    data = response.json()
    tweets = [
        {"node_id": row["row"][0], "text": row["row"][1], "tweet_id": row["row"][2]}
        for row in data["results"][0]["data"]
    ]

    if not tweets:
        # No more tweets to process
        break

    # Show progress bar, no other prints
    for tweet in tqdm(tweets, desc=f"Processing batch {skip // batch_size + 1}", unit="tweet"):
        node_id = tweet["node_id"]
        tweet_long_id = tweet["tweet_id"]
        raw_text = tweet["text"]
        text = preprocess(raw_text)

        inputs  = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)

        scores = outputs.logits[0].softmax(dim=0).numpy()
        neg, neu, pos = float(scores[0]), float(scores[1]), float(scores[2])

        sentiment_score = round(pos - neg, 4)

        max_idx = np.argmax([neg, neu, pos])
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment_label = label_map[max_idx]

        update_query = {
            "statements": [
                {
                    "statement": """
                        MATCH (t:Tweet) WHERE id(t) = $node_id
                        SET t.positive = $pos,
                            t.neutral = $neu,
                            t.negative = $neg,
                            t.sentiment_score = $score,
                            t.sentiment_label = $label,
                            t.id = $tweet_id
                    """,
                    "parameters": {
                        "node_id": node_id,
                        "pos": pos,
                        "neu": neu,
                        "neg": neg,
                        "score": sentiment_score,
                        "label": sentiment_label,
                        "tweet_id": tweet_long_id
                    }
                }
            ]
        }

        update_response = requests.post(secret.url, json=update_query, auth=auth)
        if update_response.status_code != 200:
            # Optionally handle or log errors here without printing
            pass

    skip += batch_size
