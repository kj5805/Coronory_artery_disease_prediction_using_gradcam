"""
Ensemble CAD Classification Model
Combines ResNet-18, ResNet-34, and EfficientNet-B0 for improved performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pandas as pd
import numpy as np
import cv2
import os
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# Dataset Class
# ============================================================================

class EchoCADDataset(Dataset):
    def __init__(self, csv_file, video_dir, transform=None, frame_idx=0):
        self.data = pd.read_csv(csv_file)
        self.video_dir = video_dir
        self.transform = transform
        self.frame_idx = frame_idx
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_name = row['FileName']
        label = row['CAD_Label']
        
        video_path = os.path.join(self.video_dir, video_name)
        try:
            frame = self.load_video_frame(video_path, self.frame_idx)
        except Exception as e:
            return None
        
        if self.transform:
            frame = self.transform(frame)
        
        return frame, label, video_name
    
    def load_video_frame(self, video_path, frame_idx):
        if not video_path.endswith('.avi'):
            video_path = video_path + '.avi'
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open: {video_path}")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                raise RuntimeError(f"Failed to read: {video_path}")
        
        cap.release()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

# ============================================================================
# Individual Model Definitions
# ============================================================================

class ResNet18Classifier(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18Classifier, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 2)
        
    def forward(self, x):
        return self.resnet(x)

class ResNet34Classifier(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet34Classifier, self).__init__()
        self.resnet = models.resnet34(pretrained=pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 2)
        
    def forward(self, x):
        return self.resnet(x)

class EfficientNetClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNetClassifier, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=pretrained)
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(num_features, 2)
        
    def forward(self, x):
        return self.efficientnet(x)

# ============================================================================
# Ensemble Model
# ============================================================================

class EnsembleCADClassifier(nn.Module):
    """
    Ensemble of ResNet-18, ResNet-34, and EfficientNet-B0
    Uses weighted average of predictions
    """
    
    def __init__(self, model1, model2, model3, weights=None):
        super(EnsembleCADClassifier, self).__init__()
        
        self.model1 = model1  # ResNet-18
        self.model2 = model2  # ResNet-34
        self.model3 = model3  # EfficientNet-B0
        
        # Default equal weights
        if weights is None:
            self.weights = [1/3, 1/3, 1/3]
        else:
            self.weights = weights
    
    def forward(self, x):
        # Get predictions from all models
        out1 = self.model1(x)
        out2 = self.model2(x)
        out3 = self.model3(x)
        
        # Convert to probabilities
        prob1 = torch.softmax(out1, dim=1)
        prob2 = torch.softmax(out2, dim=1)
        prob3 = torch.softmax(out3, dim=1)
        
        # Weighted average
        ensemble_prob = (self.weights[0] * prob1 + 
                        self.weights[1] * prob2 + 
                        self.weights[2] * prob3)
        
        # Convert back to logits (for loss calculation)
        ensemble_logits = torch.log(ensemble_prob + 1e-10)
        
        return ensemble_logits

# ============================================================================
# Training Functions
# ============================================================================

def skip_none_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def train_single_model(model, train_loader, val_loader, criterion, optimizer, 
                       num_epochs, device, save_path, model_name):
    """Train a single model"""
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        train_samples = 0
        
        for batch in train_loader:
            if batch is None:
                continue
            inputs, labels, _ = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_samples += inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        train_loss = train_loss / train_samples if train_samples > 0 else 0
        train_acc = accuracy_score(train_labels, train_preds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                inputs, labels, _ = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_samples += inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / val_samples if val_samples > 0 else 0
        val_acc = accuracy_score(val_labels, val_preds)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved! (Val Acc: {best_val_acc:.4f})")
    
    return model, best_val_acc

def evaluate_ensemble(ensemble, test_loader, device, save_dir):
    """Evaluate ensemble model"""
    
    ensemble.eval()
    test_preds = []
    test_labels = []
    test_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            inputs, labels, _ = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = ensemble(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_probs.extend(probs[:, 1].cpu().numpy())
    
    # Calculate metrics
    test_acc = accuracy_score(test_labels, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='binary', zero_division=0)
    auc = roc_auc_score(test_labels, test_probs)
    cm = confusion_matrix(test_labels, test_preds)
    
    print("\n" + "="*60)
    print("ENSEMBLE MODEL - TEST SET RESULTS")
    print("="*60)
    print(f"Accuracy: {test_acc:.4f} ⭐")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f} ⭐")
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Save metrics
    metrics = {
        'accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc
    }
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(save_dir, 'ensemble_metrics.csv'), index=False)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Normal', 'CAD-risk'],
                yticklabels=['Normal', 'CAD-risk'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Ensemble Model - Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'ensemble_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return metrics

def main():
    # Paths
    DATA_DIR = "C:/Users/anany/dynamic"
    CAD_CSV = "C:/Users/anany/dynamic/cad_data/CAD_FileList.csv"
    VIDEO_DIR = "C:/Users/anany/dynamic/a4c-video-dir/Videos"
    OUTPUT_DIR = "C:/Users/anany/dynamic/ensemble_output"
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == 'cpu':
        print("\n⚠️  WARNING: Training ensemble on CPU will be SLOW (~6-8 hours)")
        print("    Consider reducing num_epochs to 3-5 for faster training")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load data
    full_data = pd.read_csv(CAD_CSV)
    train_csv = full_data[full_data['Split'] == 'TRAIN']
    val_csv = full_data[full_data['Split'] == 'VAL']
    test_csv = full_data[full_data['Split'] == 'TEST']
    
    train_csv.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
    val_csv.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'), index=False)
    test_csv.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)
    
    # Create datasets
    train_dataset = EchoCADDataset(os.path.join(OUTPUT_DIR, 'train.csv'), VIDEO_DIR, transform)
    val_dataset = EchoCADDataset(os.path.join(OUTPUT_DIR, 'val.csv'), VIDEO_DIR, transform)
    test_dataset = EchoCADDataset(os.path.join(OUTPUT_DIR, 'test.csv'), VIDEO_DIR, transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, 
                             num_workers=0, collate_fn=skip_none_collate)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, 
                           num_workers=0, collate_fn=skip_none_collate)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, 
                            num_workers=0, collate_fn=skip_none_collate)
    
    print(f"\nDataset sizes:")
    print(f"Train: {len(train_dataset)}")
    print(f"Val: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")
    
    # Loss and training config
    criterion = nn.CrossEntropyLoss()
    num_epochs = 5  # Reduced for CPU training
    
    # ========================================================================
    # Train Model 1: ResNet-18
    # ========================================================================
    
    model1 = ResNet18Classifier(pretrained=True).to(device)
    optimizer1 = optim.Adam(model1.parameters(), lr=0.0001)
    
    model1, acc1 = train_single_model(
        model1, train_loader, val_loader, criterion, optimizer1,
        num_epochs, device, 
        os.path.join(OUTPUT_DIR, 'resnet18.pt'),
        "ResNet-18"
    )
    
    # ========================================================================
    # Train Model 2: ResNet-34
    # ========================================================================
    
    model2 = ResNet34Classifier(pretrained=True).to(device)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.0001)
    
    model2, acc2 = train_single_model(
        model2, train_loader, val_loader, criterion, optimizer2,
        num_epochs, device,
        os.path.join(OUTPUT_DIR, 'resnet34.pt'),
        "ResNet-34"
    )
    
    # ========================================================================
    # Train Model 3: EfficientNet-B0
    # ========================================================================
    
    model3 = EfficientNetClassifier(pretrained=True).to(device)
    optimizer3 = optim.Adam(model3.parameters(), lr=0.0001)
    
    model3, acc3 = train_single_model(
        model3, train_loader, val_loader, criterion, optimizer3,
        num_epochs, device,
        os.path.join(OUTPUT_DIR, 'efficientnet_b0.pt'),
        "EfficientNet-B0"
    )
    
    # ========================================================================
    # Create Ensemble
    # ========================================================================
    
    print("\n" + "="*60)
    print("CREATING ENSEMBLE MODEL")
    print("="*60)
    
    # Load best checkpoints
    model1.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'resnet18.pt')))
    model2.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'resnet34.pt')))
    model3.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'efficientnet_b0.pt')))
    
    # Weight models by their validation accuracy
    total_acc = acc1 + acc2 + acc3
    weights = [acc1/total_acc, acc2/total_acc, acc3/total_acc]
    
    print(f"\nModel weights (based on validation accuracy):")
    print(f"  ResNet-18: {weights[0]:.3f} (Val Acc: {acc1:.4f})")
    print(f"  ResNet-34: {weights[1]:.3f} (Val Acc: {acc2:.4f})")
    print(f"  EfficientNet-B0: {weights[2]:.3f} (Val Acc: {acc3:.4f})")
    
    # Create ensemble
    ensemble = EnsembleCADClassifier(model1, model2, model3, weights=weights)
    ensemble = ensemble.to(device)
    
    # Evaluate ensemble
    metrics = evaluate_ensemble(ensemble, test_loader, device, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("ENSEMBLE TRAINING COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nExpected improvements over single model:")
    print("  ✅ 2-5% better accuracy")
    print("  ✅ More robust predictions")
    print("  ✅ Better generalization")
    print("\nGenerated files:")
    print("  - resnet18.pt")
    print("  - resnet34.pt")
    print("  - efficientnet_b0.pt")
    print("  - ensemble_metrics.csv")
    print("  - ensemble_confusion_matrix.png")

if __name__ == "__main__":
    main()
