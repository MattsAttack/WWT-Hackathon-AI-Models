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


class Prediction(BaseModel):
    result: Literal["Phishing Email", "Legitimate Email"]


def predict_email(email_subject: str, email_body: str) -> Prediction:
    """Predict whether an email is phishing or legitimate using the trained BERT model.

    :param email_subject: Subject of the email.
    :param email_body: Body of the email.

    :return: One of "Phishing Email" or "Legitimate Email".
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


if __name__ == "__main__":
    # Example email data
    email_1_subject = "Update: New Polling Locations Announced"
    email_1_body = "QUICK, you've fallen behind on your student loans. Click here to pay them at a discounted rate"

    # Predict whether the email is phishing
    print("starting prediction 1")
    result_1 = predict_email(email_1_subject, email_1_body)
    print(result_1.result)  # Output: "Phishing Email"

    email_2_subject = "Lunch"
    email_2_body = "Let's go get lunch, im hungry"

    # Predict whether the email is phishing
    print("starting prediction 2")
    result_2 = predict_email(email_2_subject, email_2_body)
    print(result_2.result)  # Output: "Legitimate Email"
