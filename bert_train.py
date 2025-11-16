# train_consistent_bert_nn.py
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
from tqdm import tqdm
import numpy as np

# Consistent model name - use the same everywhere
MODEL_NAME = "prajjwal1/bert-mini"  # Good balance of speed and performance

class EnhancedBERTClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, n_classes=5, dropout_rate=0.4):
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

class ResumeDataset(Dataset):
    def __init__(self, job_descriptions, resumes, labels, tokenizer, max_length=256):
        self.job_descriptions = job_descriptions
        self.resumes = resumes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.job_descriptions)
    
    def __getitem__(self, idx):
        job_desc = str(self.job_descriptions.iloc[idx])
        resume = str(self.resumes.iloc[idx])
        
        # Enhanced text combination
        text = f"Job Requirements: {job_desc} Candidate Qualifications: {resume}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels.iloc[idx] - 1, dtype=torch.long)
        }

def analyze_dataset(df):
    """Analyze dataset characteristics"""
    print("\n📊 DATASET ANALYSIS:")
    print(f"Total samples: {len(df)}")
    print(f"Class distribution:\n{df['match_score'].value_counts().sort_index()}")
    
    # Text length analysis
    df['job_len'] = df['job_description'].str.len()
    df['resume_len'] = df['resume'].str.len()
    
    print(f"Average job description length: {df['job_len'].mean():.0f} chars")
    print(f"Average resume length: {df['resume_len'].mean():.0f} chars")
    print(f"Shortest job: {df['job_len'].min()} chars, Longest job: {df['job_len'].max()} chars")
    print(f"Shortest resume: {df['resume_len'].min()} chars, Longest resume: {df['resume_len'].max()} chars")

def train_enhanced_bert_model(csv_file_path, sample_size=None):
    print("🚀 Loading data...")
    df = pd.read_csv(csv_file_path)
    
    # Data cleaning
    initial_size = len(df)
    df = df.dropna(subset=['job_description', 'resume', 'match_score'])
    df = df[df['job_description'].str.len() > 50]
    df = df[df['resume'].str.len() > 50]
    
    print(f"🧹 Data cleaning: {initial_size} → {len(df)} samples")
    
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
        print(f"Using {sample_size} samples")
    else:
        print(f"Using full dataset: {len(df)} samples")
    
    analyze_dataset(df)
    
    # Initialize with the same model name
    print(f"🎯 Loading {MODEL_NAME} with Neural Network head...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EnhancedBERTClassifier(model_name=MODEL_NAME, n_classes=5)
    
    # Train-validation-test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        df[['job_description', 'resume']], 
        df['match_score'],
        test_size=0.15, 
        random_state=42,
        stratify=df['match_score']
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.15, 
        random_state=42,
        stratify=y_temp
    )
    
    print(f"📚 Training samples: {len(X_train)}")
    print(f"📋 Validation samples: {len(X_val)}")
    print(f"🧪 Test samples: {len(X_test)}")
    
    # Datasets
    train_dataset = ResumeDataset(X_train['job_description'], X_train['resume'], y_train, tokenizer)
    val_dataset = ResumeDataset(X_val['job_description'], X_val['resume'], y_val, tokenizer)
    test_dataset = ResumeDataset(X_test['job_description'], X_test['resume'], y_test, tokenizer)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Using device: {device}")
    model = model.to(device)
    
    # Handle class imbalance
    labels_for_weights = y_train.values - 1  # Convert to 0-based for class weights
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(labels_for_weights), 
        y=labels_for_weights
    )
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"⚖️ Class weights: {class_weights}")
    
    # Enhanced optimizer and loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # Fixed learning rate scheduler (removed verbose parameter)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5
    )
    
    # Training with early stopping
    print("🎯 Starting enhanced training...")
    best_accuracy = 0
    patience = 8
    patience_counter = 0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(5):  # Reduced epochs for faster training
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/5'):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        predictions = []
        actual_labels = []
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().tolist())
                actual_labels.extend(labels.cpu().tolist())
        
        val_accuracy = accuracy_score(actual_labels, predictions)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracies.append(val_accuracy)
        train_losses.append(total_loss / len(train_loader))
        
        # Learning rate scheduling
        scheduler.step(val_accuracy)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch {epoch+1}:')
        print(f'  Train Loss: {total_loss/len(train_loader):.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Val Accuracy: {val_accuracy:.4f}')
        print(f'  Learning Rate: {current_lr:.2e}')
        
        # Early stopping and model saving
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'enhanced_bert_nn_model.pth')
            patience_counter = 0
            print(f'  → New best model saved with accuracy: {val_accuracy:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load('enhanced_bert_nn_model.pth'))
    model.eval()
    
    # Final evaluation on test set
    final_predictions = []
    final_actual_labels = []
    final_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)
            
            final_predictions.extend(preds.cpu().tolist())
            final_actual_labels.extend(labels.cpu().tolist())
            final_probabilities.extend(probabilities.cpu().tolist())
    
    # Convert back to 1-5 scale
    final_predictions_original = [p + 1 for p in final_predictions]
    final_actual_labels_original = [l + 1 for l in final_actual_labels]
    
    final_accuracy = accuracy_score(final_actual_labels_original, final_predictions_original)
    
    print("\n" + "="*60)
    print("🎯 FINAL ENHANCED BERT + NEURAL NETWORK RESULTS")
    print("="*60)
    print(f"🏆 Best Validation Accuracy: {best_accuracy:.4f}")
    print(f"📊 Final Test Accuracy: {final_accuracy:.4f}")
    print("\n📈 Classification Report:")
    print(classification_report(final_actual_labels_original, final_predictions_original, 
                              target_names=['Score 1', 'Score 2', 'Score 3', 'Score 4', 'Score 5']))
    
    # Calculate per-class accuracy
    print("\n📊 Per-class Accuracy:")
    for i in range(1, 6):
        class_mask = np.array(final_actual_labels_original) == i
        if class_mask.sum() > 0:
            class_acc = accuracy_score(
                np.array(final_actual_labels_original)[class_mask],
                np.array(final_predictions_original)[class_mask]
            )
            print(f"  Score {i}: {class_acc:.3f} ({class_mask.sum()} samples)")
    
    # Save everything
    joblib.dump(tokenizer, 'enhanced_bert_tokenizer.pkl')
    
    training_info = {
        'model_name': MODEL_NAME,
        'dataset_size': len(df),
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'best_validation_accuracy': best_accuracy,
        'final_test_accuracy': final_accuracy,
        'hidden_size': model.bert.config.hidden_size,
        'architecture': 'BERT + Enhanced Neural Network',
        'class_weights': class_weights.cpu().numpy().tolist()
    }
    joblib.dump(training_info, 'enhanced_bert_info.pkl')
    
    print(f"\n✅ Enhanced model trained successfully with {MODEL_NAME}")
    print(f"🔧 Architecture: BERT + Neural Network")
    print(f"📐 Hidden size: {model.bert.config.hidden_size}")
    print(f"📊 Trained on {len(X_train)} samples")
    
    return final_accuracy

# Alternative: Simple version without learning rate scheduler
def train_simple_bert_model(csv_file_path):
    """Simplified version without advanced features"""
    print("🚀 Loading data for simple training...")
    df = pd.read_csv(csv_file_path)
    
    # Basic cleaning
    df = df.dropna(subset=['job_description', 'resume', 'match_score'])
    print(f"Using {len(df)} samples")
    
    # Initialize model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EnhancedBERTClassifier(model_name=MODEL_NAME, n_classes=5)
    
    # Simple train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df[['job_description', 'resume']], 
        df['match_score'],
        test_size=0.2, 
        random_state=42,
        stratify=df['match_score']
    )
    
    # Datasets
    train_dataset = ResumeDataset(X_train['job_description'], X_train['resume'], y_train, tokenizer)
    test_dataset = ResumeDataset(X_test['job_description'], X_test['resume'], y_test, tokenizer)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # Simple training loop
    print("🎯 Starting simple training...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/5'):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}: Loss: {total_loss/len(train_loader):.4f}')
    
    # Save model
    torch.save(model.state_dict(), 'simple_bert_nn_model.pth')
    joblib.dump(tokenizer, 'simple_bert_tokenizer.pkl')
    print("✅ Simple model training completed!")

if __name__ == '__main__':
    csv_file_path = 'resume_job_matching_dataset.csv'
    
    try:
        # Train enhanced model
        accuracy = train_enhanced_bert_model(csv_file_path)
        print(f"\n🎯 Final Enhanced Model Accuracy: {accuracy:.4f}")
        
    except Exception as e:
        print(f"❌ Error in enhanced training: {e}")
        print("🔄 Trying simple training...")
        train_simple_bert_model(csv_file_path)