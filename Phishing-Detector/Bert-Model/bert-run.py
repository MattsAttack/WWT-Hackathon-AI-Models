import pathlib
from typing import Literal

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

p = pathlib.Path(__file__).parent.resolve().parent

# Load the trained model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained(p / "trained_model")
tokenizer = DistilBertTokenizer.from_pretrained(p / "trained_model")


def predict_email(
    email_subject: str,
    email_body: str,
) -> Literal["Phishing Email", "Legitimate Email"]:
    """Predict whether an email is phishing or legitimate using the trained BERT model.


    :param email_subject: Subject of the email.
    :param email_body: Body of the email.

    :return: One of "Phishing Email" or "Legitimate Email".
    """
    # Combine subject and body into one text
    email_text = f"{email_subject} {email_body}"

    # Tokenize the text
    inputs = tokenizer(
        email_text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(
            outputs.logits, dim=1
        ).item()  # 0 = legitimate, 1 = phishing

    # Return result
    return "Phishing Email" if prediction == 1 else "Legitimate Email"


# Example email data
# email_subject = "Update: New Polling Locations Announced"
# email_body = "QUICK, you've fallen behind on your student loans. Click here to pay them at a discounted rate"
email_subject = "Lunch"
email_body = "Let's go get lunch, im hungry"

# Predict whether the email is phishing
print("starting prediction")
result = predict_email(email_subject, email_body)
print(result)  # Output: "Phishing Email"
