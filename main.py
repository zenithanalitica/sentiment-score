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
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    scores = outputs.logits[0].softmax(dim=0).numpy()
    labels = Labels(
        float(scores[0]),
        float(scores[1]),
        float(scores[2]),
    )

    sentiment_score = round(labels.pos - labels.neg, 4)
    return (labels, sentiment_score)


def update_sentiment(
    driver: neo4j.Driver, tweet: Tweet, labels: Labels, sentiment_score: float
) -> None:
    # Find the maximum value and set sentiment label directly
    values = {
        "positive": labels.pos,
        "neutral": labels.neu,
        "negative": labels.neg,
    }
    sentiment_label = cast(str, max(values, key=values.get))  # pyright: ignore[reportArgumentType, reportCallIssue]

    update_query: LiteralString = """
            MATCH (t:Tweet) WHERE elementId(t) = $node_id
            SET t.positive = $pos,
                t.neutral = $neu,
                t.negative = $neg,
                t.sentiment_score = $score,
                t.sentiment_label = $label,
                t.id = $tweet_id
        """

    _ = driver.execute_query(
        update_query,
        node_id=tweet.node_id,
        pos=labels.pos,  # Fixed variable name from label to labels
        neu=labels.neu,
        neg=labels.neg,
        score=sentiment_score,
        label=sentiment_label,
        tweet_id=tweet.tweet_id,
    )


def main() -> None:
    batch_size = 1000
    skip = 0

    # ————— Load model & tokenizer —————
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    # Explicitly set use_fast=False and legacy=True to avoid tiktoken compatibility issues in Python 3.13
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    model = AutoModelForSequenceClassification.from_pretrained(model_name)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

    # ————— Neo4j HTTP setup —————
    with GraphDatabase.driver(secret.url, auth=("neo4j", secret.password)) as driver:
        driver.verify_connectivity()
        assert driver.verify_authentication()

        while True:
            tweets = get_tweets(driver, skip, batch_size)
            if not tweets:
                # No more tweets to process
                break

            # Show progress bar, no other prints
            for tweet in tqdm(
                tweets, desc=f"Processing batch {skip // batch_size + 1}", unit="tweet"
            ):
                text = preprocess(tweet.text)
                labels, sentiment_score = calculate_scores(model, tokenizer, text)
                update_sentiment(driver, tweet, labels, sentiment_score)

            skip += batch_size


if __name__ == "__main__":
    main()
