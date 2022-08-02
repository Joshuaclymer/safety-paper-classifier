from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
import torch
import pandas as pd
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)
model.load_state_dict(torch.load(r"C:\Users\joshu\GitHub\safety-paper-classifier\DistillBERT\results\pytorch_model.bin"))

train = pd.read_csv('../Data/Processed/train.csv')
test = pd.read_csv('../Data/Processed/test.csv')

# Preprocessing
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
inputs = tokenizer(list(test["text"][0:1]), return_tensors="pt", padding=True)
print(model(**inputs))
