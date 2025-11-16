from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json

# Download stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)

# Custom JSON encoder to handle numpy types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)

app.json_encoder = CustomJSONEncoder

# Load TF-IDF + Random Forest models
try:
    tfidf_model = joblib.load('random_forest.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("✅ TF-IDF + Random Forest model loaded successfully")
except Exception as e:
    print(f"❌ Error loading TF-IDF model: {str(e)}")
    tfidf_model = None
    tfidf_vectorizer = None

# BERT + Neural Network Model Class
class EnhancedBERTClassifier(nn.Module):
    def __init__(self, model_name='prajjwal1/bert-mini', n_classes=5, dropout_rate=0.4):
        super(EnhancedBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Enhanced Neural Network head
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(64, n_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Pass through enhanced classifier
        logits = self.classifier(pooled_output)
        return logits

# Load BERT + Neural Network model
try:
    bert_training_info = joblib.load('enhanced_bert_info.pkl')
    MODEL_NAME = bert_training_info.get('model_name', 'prajjwal1/bert-mini')
    
    bert_tokenizer = joblib.load('enhanced_bert_tokenizer.pkl')
    bert_model = EnhancedBERTClassifier(model_name=MODEL_NAME, n_classes=5)
    bert_model.load_state_dict(torch.load('enhanced_bert_nn_model.pth', map_location=torch.device('cpu')))
    bert_model.eval()
    
    print("✅ BERT + Neural Network model loaded successfully!")
    print(f"📊 BERT trained on {bert_training_info.get('training_samples', 'N/A')} samples")
    print(f"🎯 BERT accuracy: {bert_training_info.get('final_test_accuracy', 'N/A')}")
    
except Exception as e:
    print(f"❌ Error loading BERT + Neural Network model: {str(e)}")
    bert_tokenizer = None
    bert_model = None
    bert_training_info = {}

# Define stopwords
stop_words = set(stopwords.words("english"))

def clean_text(text):
    """Clean and preprocess text data"""
    if pd.isnull(text) or text == "":
        return ""
    
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def train_tfidf_vectorizer(resumes, max_features=800):
    """Train and save TF-IDF vectorizer on resume data"""
    # Clean resumes
    cleaned_resumes = [clean_text(resume) for resume in resumes]
    
    # Create and fit TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    X_tfidf = tfidf_vectorizer.fit_transform(cleaned_resumes)
    
    # Save the vectorizer
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
    print("TF-IDF vectorizer trained and saved successfully")
    
    return tfidf_vectorizer, X_tfidf

def predict_tfidf_match(resume_text, job_description):
    """Predict match score using TF-IDF + Random Forest"""
    if not tfidf_vectorizer or not tfidf_model:
        return None, None, None
    
    try:
        cleaned_resume = clean_text(resume_text)
        cleaned_jd = clean_text(job_description)
        combined_text = cleaned_resume + " " + cleaned_jd
        
        text_vectorized = tfidf_vectorizer.transform([combined_text])
        prediction = tfidf_model.predict(text_vectorized)[0]
        probability = tfidf_model.predict_proba(text_vectorized)[0]
        confidence = float(max(probability))
        
        is_shortlisted = bool(prediction >= 4)
        return is_shortlisted, confidence, int(prediction)
    except Exception as e:
        print(f"TF-IDF prediction error: {e}")
        return None, None, None

def predict_bert_match(resume_text, job_description):
    """Predict match score using BERT + Neural Network"""
    if not bert_tokenizer or not bert_model:
        return None, None, None
    
    try:
        # Prepare text
        job_desc_short = str(job_description)[:800]
        resume_short = str(resume_text)[:800]
        text = f"Job Requirements: {job_desc_short} Candidate Qualifications: {resume_short}"
        
        # Tokenize
        encoding = bert_tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )
        
        # Predict with neural network
        with torch.no_grad():
            outputs = bert_model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask']
            )
            probabilities = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
            
            # Convert back to 1-5 scale and ensure native Python types
            prediction = int(prediction.item() + 1)
            confidence = float(confidence.item())
            
            is_shortlisted = bool(prediction >= 4)
            return is_shortlisted, confidence, prediction
            
    except Exception as e:
        print(f"BERT prediction error: {e}")
        return None, None, None

@app.route('/')
def index():
    tfidf_status = "loaded" if tfidf_model and tfidf_vectorizer else "not loaded"
    bert_status = "loaded" if bert_model and bert_tokenizer else "not loaded"
    bert_accuracy = bert_training_info.get('final_test_accuracy', 'N/A') if bert_training_info else 'N/A'
    
    return render_template('index.html', 
                         tfidf_status=tfidf_status, 
                         bert_status=bert_status,
                         bert_accuracy=bert_accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        resume_text = request.form.get('resume_text', '')
        job_description = request.form.get('job_description', '')
        
        if not resume_text or not job_description:
            return jsonify({
                'error': 'Please provide both resume text and job description'
            }), 400
        
        # Get predictions from both models
        tfidf_shortlisted, tfidf_confidence, tfidf_score = predict_tfidf_match(resume_text, job_description)
        bert_shortlisted, bert_confidence, bert_score = predict_bert_match(resume_text, job_description)
        
        # Check if models agree
        models_agree = None
        if tfidf_shortlisted is not None and bert_shortlisted is not None:
            models_agree = bool(tfidf_shortlisted == bert_shortlisted)
        
        # Prepare response
        response = {
            'models': {
                'tfidf': {
                    'name': 'TF-IDF + Random Forest',
                    'result': "Shortlisted" if tfidf_shortlisted else "Not Shortlisted" if tfidf_shortlisted is not None else "Model Error",
                    'confidence': round(float(tfidf_confidence or 0) * 100, 2),
                    'raw_score': int(tfidf_score or 0),
                    'status': 'success' if tfidf_shortlisted is not None else 'error'
                },
                'bert': {
                    'name': 'BERT + Neural Network',
                    'result': "Shortlisted" if bert_shortlisted else "Not Shortlisted" if bert_shortlisted is not None else "Model Error",
                    'confidence': round(float(bert_confidence or 0) * 100, 2),
                    'raw_score': int(bert_score or 0),
                    'status': 'success' if bert_shortlisted is not None else 'error'
                }
            },
            'summary': {
                'resume_length': len(resume_text.split()),
                'jd_length': len(job_description.split()),
                'models_agree': models_agree,
                'agreement_text': "✅ Models Agree" if models_agree else "⚠️ Models Disagree" if models_agree is not None else "❓ No Comparison"
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload OR text input for resume"""
    try:
        job_description = request.form.get('job_description', '')
        resume_file = request.files.get('resume_file')
        resume_text = request.form.get('resume_text', '')
        
        # Check if we have job description
        if not job_description:
            return jsonify({'error': 'Please provide job description'}), 400
        
        # Check if we have either resume text OR resume file
        if not resume_text.strip() and (not resume_file or resume_file.filename == ''):
            return jsonify({'error': 'Please provide either resume text or upload a resume file'}), 400
        
        # Process resume content
        if resume_file and resume_file.filename != '':
            # Handle file upload
            if resume_file.filename.lower().endswith('.txt'):
                resume_text_content = resume_file.read().decode('utf-8')
            elif resume_file.filename.lower().endswith('.pdf'):
                try:
                    import PyPDF2
                    pdf_reader = PyPDF2.PdfReader(resume_file)
                    resume_text_content = ""
                    for page in pdf_reader.pages:
                        resume_text_content += page.extract_text() + " "
                except ImportError:
                    return jsonify({
                        'error': 'PDF processing requires PyPDF2. Install with: pip install PyPDF2'
                    }), 500
                except Exception as e:
                    return jsonify({
                        'error': f'Error reading PDF: {str(e)}'
                    }), 500
            else:
                return jsonify({
                    'error': 'Unsupported file type. Please upload .txt or .pdf files.'
                }), 400
        else:
            # Use the text from textarea
            resume_text_content = resume_text
        
        # Get predictions from both models using the resume content
        tfidf_shortlisted, tfidf_confidence, tfidf_score = predict_tfidf_match(resume_text_content, job_description)
        bert_shortlisted, bert_confidence, bert_score = predict_bert_match(resume_text_content, job_description)
        
        # Check if models agree
        models_agree = None
        if tfidf_shortlisted is not None and bert_shortlisted is not None:
            models_agree = bool(tfidf_shortlisted == bert_shortlisted)
        
        response = {
            'models': {
                'tfidf': {
                    'name': 'TF-IDF + Random Forest',
                    'result': "Shortlisted" if tfidf_shortlisted else "Not Shortlisted" if tfidf_shortlisted is not None else "Model Error",
                    'confidence': round(float(tfidf_confidence or 0) * 100, 2),
                    'raw_score': int(tfidf_score or 0),
                    'status': 'success' if tfidf_shortlisted is not None else 'error'
                },
                'bert': {
                    'name': 'BERT + Neural Network',
                    'result': "Shortlisted" if bert_shortlisted else "Not Shortlisted" if bert_shortlisted is not None else "Model Error",
                    'confidence': round(float(bert_confidence or 0) * 100, 2),
                    'raw_score': int(bert_score or 0),
                    'status': 'success' if bert_shortlisted is not None else 'error'
                }
            },
            'summary': {
                'resume_length': len(resume_text_content.split()),
                'jd_length': len(job_description.split()),
                'models_agree': models_agree,
                'agreement_text': "✅ Models Agree" if models_agree else "⚠️ Models Disagree" if models_agree is not None else "❓ No Comparison",
                'file_processed': bool(resume_file and resume_file.filename != '')
            }
        }
        
        return jsonify(response)
            
    except Exception as e:
        return jsonify({
            'error': f'File processing failed: {str(e)}'
        }), 500

@app.route('/train_vectorizer', methods=['POST'])
def train_vectorizer():
    """Endpoint to train TF-IDF vectorizer on new data"""
    try:
        # This would typically receive training data
        data = request.get_json()
        
        if not data or 'resumes' not in data:
            return jsonify({
                'error': 'Please provide resume data for training'
            }), 400
        
        resumes = data['resumes']
        vectorizer, X_tfidf = train_tfidf_vectorizer(resumes)
        
        return jsonify({
            'message': 'TF-IDF vectorizer trained successfully',
            'features_shape': X_tfidf.shape
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Vectorizer training failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)