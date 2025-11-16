# train_model.py
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Download stopwords
nltk.download('stopwords')

# Load your dataset
df = pd.read_csv('resume_job_matching_dataset.csv')

def clean_text(text):
    if pd.isnull(text) or text == "":
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Preprocess data
stop_words = set(stopwords.words('english'))
df['resume_clean'] = df['resume'].apply(clean_text)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=800)
X_tfidf = tfidf_vectorizer.fit_transform(df['resume_clean'])

# Target variable (adjust based on your dataset)
y = df['match_score']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'random_forest.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

print("Model and vectorizer saved successfully!")