from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from datasets import Dataset
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.special import softmax

# === Paths ===
input_path = "/kaggle/input/split/full_length_tweets_part_1.pkl"
output_path_pkl = "/kaggle/working/bert_tweets_classified_part_1.pkl"
output_path_csv = "/kaggle/working/bert_tweets_classified_part_1.csv"

# === Load and clean data ===
df = pd.read_pickle(input_path)
df = df[df["text"].apply(lambda x: isinstance(x, str))].reset_index(drop=True)

df = df.iloc[:500].copy()

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df[["text"]])

# === Load model, tokenizer, and config ===
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to("cuda")
model.eval()


# === Preprocess function ===
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


df["text"] = df["text"].apply(preprocess)


# === Tokenization ===
def tokenize(batch):
    return tokenizer(
        batch["text"], padding="max_length", truncation=True, max_length=256
    )


tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
dataloader = DataLoader(tokenized_dataset, batch_size=128)

# === Inference ===
all_preds = []
all_scores = []

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Classifying"):
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.cpu().numpy()
        probs = softmax(logits, axis=1)  # shape (batch_size, num_classes)
        preds = np.argmax(probs, axis=1)

        all_preds.extend(preds)
        all_scores.extend(probs)

# === Add predictions and scores to DataFrame ===
label_map = config.id2label
df["sentiment"] = [label_map[p] for p in all_preds]
df["sentiment_scores"] = all_scores  # This will be a column of arrays

# Optional: Split scores into separate columns
score_labels = [label_map[i] for i in range(len(label_map))]
score_array = np.array(all_scores)
for i, label in enumerate(score_labels):
    df[f"score_{label.lower()}"] = score_array[:, i]

# === Save Results ===
df.to_csv(output_path_csv, index=False)
df.to_pickle(output_path_pkl)
