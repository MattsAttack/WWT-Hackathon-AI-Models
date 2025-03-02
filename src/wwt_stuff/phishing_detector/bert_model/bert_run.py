from pathlib import Path
from typing import Literal

from pydantic import BaseModel
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

p: Path = Path(__file__).parent.resolve()
model_path = p / "trained_model"

# Load the trained model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)


type PredictionResult = Literal["Phishing Email", "Legitimate Email"]


class Prediction(BaseModel):
    result: PredictionResult


def predict_email(email_subject: str, email_body: str) -> Prediction:
    """Predict whether an email is phishing or legitimate using the trained BERT model.

    :param email_subject: Subject of the email.
    :param email_body: Body of the email.

    :return: Prediction with result as one of "Phishing Email" or "Legitimate Email".
    """
    # Combine subject and body into one text
    email_text = f"{email_subject} {email_body}"

    # Tokenize the text
    inputs = tokenizer(
        email_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(
            outputs.logits,
            dim=1,
        ).item()  # 0 = legitimate, 1 = phishing

    # Return result
    return Prediction(
        result="Phishing Email" if prediction == 1 else "Legitimate Email"
    )
