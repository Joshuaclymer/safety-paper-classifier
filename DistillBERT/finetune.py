#https://huggingface.co/docs/transformers/model_doc/distilbert#transformers.DistilBertForSequenceClassification.forward.example
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

train = pd.read_csv('../Data/Processed/train.csv')
test = pd.read_csv('../Data/Processed/test.csv')

# Preprocessing
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

trainDataset = Dataset(tokenizer(list(train["text"]), truncation=True), list(train["label"]))
testDataset = Dataset(tokenizer(list(test["text"]), truncation=True), list(test["label"]))

#fine-tune
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=trainDataset,
    eval_dataset=testDataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()