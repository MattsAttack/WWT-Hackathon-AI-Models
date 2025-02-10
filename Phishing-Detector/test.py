import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset (update the path if needed)
df = pd.read_csv(
    "c:/Users/matts/Desktop/Code-Shit/Python-Stuff/WWT-Stuff/Phishing-Detector/CEAS_08.csv",
    encoding="latin1",
)

# Combine subject and body into a single text column
df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")

# Select relevant columns
df = df[["text", "label"]]


# Text Cleaning Function
def clean_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )  # Remove punctuation
    return text


# Apply text cleaning
df["cleaned_text"] = df["text"].apply(clean_text)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned_text"], df["label"], test_size=0.2, random_state=42
)

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predict on test data
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# Function to classify a new email
def predict_email(email_subject, email_body):
    email_text = email_subject + " " + email_body
    cleaned_email = clean_text(email_text)
    email_tfidf = vectorizer.transform([cleaned_email])
    prediction = model.predict(email_tfidf)
    return "Phishing Email" if prediction[0] == 1 else "Legitimate Email"


# Example usage
new_subject = "Lunch Tomorrow"
new_body = "Hey greg, want to get lunch tomorrow?"
print("Prediction:", predict_email(new_subject, new_body))
