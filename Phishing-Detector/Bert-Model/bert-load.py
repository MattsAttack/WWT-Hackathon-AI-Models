import pandas as pd
import torch
from transformers import DistilBertTokenizer  # spell-checker: disable
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)  # spell-checker: disable
from transformers import TrainerCallback

# Make sure to run pip install transformers torch datasets scikit-learn


# Load dataset
df = pd.read_csv(
    "c:/Users/matts/Desktop/Code-Shit/Python-Stuff/WWT-Stuff/Phishing-Detector/CEAS_08.csv",
    encoding="latin1",
)

# Combine subject and body
df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")  # noqa: fillna is a valid pandas method

# Convert labels: 1 = Phishing, 0 = Legitimate
df["label"] = df["label"].astype(int)

# Split dataset into train/test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

# Load tokenizer for DistilBERT  # spell-checker: disable
tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased"
)  # spell-checker: disable

# Tokenize text
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)


# Convert into PyTorch Dataset
class EmailDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


train_dataset = EmailDataset(train_encodings, train_labels)
test_dataset = EmailDataset(test_encodings, test_labels)


model = DistilBertForSequenceClassification.from_pretrained(  # spell-checker: disable
    "distilbert-base-uncased", num_labels=2
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

print("Implementing call back")


class TrainingStatusMonitor(TrainerCallback):
    def on_log(self, _args, state, _control, logs=None, **_kwargs):
        if logs is not None:
            print(f"Step {state.global_step}: {logs}")


trainer.add_callback(TrainingStatusMonitor)
print("about to train")
trainer.train()

# Save the trained model and tokenizer
trainer.save_model("./trained_model")
tokenizer.save_pretrained("./trained_model")
