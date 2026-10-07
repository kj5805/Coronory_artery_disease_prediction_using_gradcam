"""
Temporal CAD Classification using Multiple Frames
Analyzes cardiac motion across time by using sequence of frames
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
# Multi-Frame Dataset
# ============================================================================

class MultiFrameEchoDataset(Dataset):
    """
    Dataset that extracts multiple frames from each video
    Captures temporal information across cardiac cycle
    """
    
    def __init__(self, csv_file, video_dir, transform=None, num_frames=5):
        """
        Args:
            num_frames: Number of frames to extract from each video
                       Evenly spaced across the video
        """
        self.data = pd.read_csv(csv_file)
        self.video_dir = video_dir
        self.transform = transform
        self.num_frames = num_frames
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_name = row['FileName']
        label = row['CAD_Label']
        
        video_path = os.path.join(self.video_dir, video_name)
        try:
            frames = self.load_video_frames(video_path, self.num_frames)
        except Exception as e:
            return None
        
        if self.transform:
            frames = torch.stack([self.transform(frame) for frame in frames])
        
        return frames, label, video_name
    
    def load_video_frames(self, video_path, num_frames):
        """Extract multiple frames evenly spaced across video"""
        if not video_path.endswith('.avi'):
            video_path = video_path + '.avi'
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open: {video_path}")
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < num_frames:
            # If video has fewer frames than requested, use available frames
            frame_indices = list(range(total_frames))
        else:
            # Evenly space frames across video
            frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise RuntimeError(f"No frames extracted from: {video_path}")
        
        # If we got fewer frames than requested, duplicate last frame
        while len(frames) < num_frames:
            frames.append(frames[-1].copy())
        
        return frames

# ============================================================================
# Temporal Model Architecture
# ============================================================================

class TemporalCADClassifier(nn.Module):
    """
    Temporal model using LSTM to process sequence of CNN features
    Architecture: ResNet-18 (feature extractor) + LSTM (temporal) + FC (classifier)
    """
    
    def __init__(self, num_frames=5, pretrained=True):
        super(TemporalCADClassifier, self).__init__()
        
        # Feature extractor (ResNet-18 without final FC)
        resnet = models.resnet18(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze feature extractor (optional - saves training time)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=512,  # ResNet-18 final layer outputs 512 features
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, num_frames, channels, height, width]
        """
        batch_size, num_frames, c, h, w = x.size()
        
        # Extract features from each frame
        # Reshape: [batch_size * num_frames, channels, height, width]
        x = x.view(batch_size * num_frames, c, h, w)
        
        # Extract features
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Reshape back to sequence: [batch_size, num_frames, feature_dim]
        features = features.view(batch_size, num_frames, -1)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(features)
        
        # Use final hidden state for classification
        final_hidden = hidden[-1]  # Last layer's final hidden state
        
        # Classify
        output = self.fc(final_hidden)
        
        return output

# ============================================================================
# Alternative: 3D CNN Model
# ============================================================================

class CNN3DClassifier(nn.Module):
    """
    3D CNN for spatiotemporal feature learning
    Processes volume of frames directly
    """
    
    def __init__(self, num_frames=5):
        super(CNN3DClassifier, self).__init__()
        
        self.conv3d_layers = nn.Sequential(
            # Input: [batch, 3, num_frames, 224, 224]
            nn.Conv3d(3, 32, kernel_size=(3, 7, 7), padding=(1, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 7, 7))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch, num_frames, channels, height, width]
        """
        # Reshape to [batch, channels, num_frames, height, width] for 3D conv
        x = x.permute(0, 2, 1, 3, 4)
        
        x = self.conv3d_layers(x)
        x = self.classifier(x)
        
        return x

# ============================================================================
# Training Functions
# ============================================================================

def skip_none_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def train_temporal_model(model, train_loader, val_loader, criterion, optimizer, 
                        num_epochs, device, save_dir, model_name):
    """Train temporal model"""
    
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
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
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
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
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, f'{model_name}.pt'))
            print(f"✅ Best model saved! (Val Acc: {best_val_acc:.4f})")
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accs, label='Train Acc', marker='o')
    ax2.plot(val_accs, label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{model_name} - Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return model

def evaluate_temporal_model(model, test_loader, device, save_dir, model_name):
    """Evaluate temporal model"""
    
    model.eval()
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
            
            outputs = model(inputs)
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
    print(f"{model_name.upper()} - TEST SET RESULTS")
    print("="*60)
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Save metrics
    metrics = {
        'model': model_name,
        'accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc
    }
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(save_dir, f'{model_name}_metrics.csv'), index=False)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                xticklabels=['Normal', 'CAD-risk'],
                yticklabels=['Normal', 'CAD-risk'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return metrics

def main():
    # Paths
    DATA_DIR = "C:/Users/anany/dynamic"
    CAD_CSV = "C:/Users/anany/dynamic/cad_data/CAD_FileList.csv"
    VIDEO_DIR = "C:/Users/anany/dynamic/a4c-video-dir/Videos"
    OUTPUT_DIR = "C:/Users/anany/dynamic/temporal_output"
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == 'cpu':
        print("\n⚠️  WARNING: Training temporal models on CPU will be SLOW")
        print("    Recommended: Reduce num_epochs to 3 and num_frames to 3")
    
    # Configuration
    NUM_FRAMES = 5  # Number of frames per video
    NUM_EPOCHS = 5  # Reduced for CPU
    BATCH_SIZE = 8  # Smaller batch for temporal models
    
    print(f"\nConfiguration:")
    print(f"  Frames per video: {NUM_FRAMES}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    
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
    
    # Create multi-frame datasets
    train_dataset = MultiFrameEchoDataset(os.path.join(OUTPUT_DIR, 'train.csv'), 
                                         VIDEO_DIR, transform, NUM_FRAMES)
    val_dataset = MultiFrameEchoDataset(os.path.join(OUTPUT_DIR, 'val.csv'), 
                                       VIDEO_DIR, transform, NUM_FRAMES)
    test_dataset = MultiFrameEchoDataset(os.path.join(OUTPUT_DIR, 'test.csv'), 
                                        VIDEO_DIR, transform, NUM_FRAMES)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=0, collate_fn=skip_none_collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=0, collate_fn=skip_none_collate)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=0, collate_fn=skip_none_collate)
    
    print(f"\nDataset sizes:")
    print(f"Train: {len(train_dataset)}")
    print(f"Val: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")
    
    # ========================================================================
    # Train LSTM-based Temporal Model
    # ========================================================================
    
    print("\n" + "="*60)
    print("TRAINING LSTM-BASED TEMPORAL MODEL")
    print("="*60)
    
    lstm_model = TemporalCADClassifier(num_frames=NUM_FRAMES, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.0001)
    
    lstm_model = train_temporal_model(
        lstm_model, train_loader, val_loader, criterion, optimizer,
        NUM_EPOCHS, device, OUTPUT_DIR, "temporal_lstm"
    )
    
    # Evaluate
    lstm_model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'temporal_lstm.pt')))
    lstm_metrics = evaluate_temporal_model(lstm_model, test_loader, device, OUTPUT_DIR, "temporal_lstm")
    
    print("\n" + "="*60)
    print("TEMPORAL TRAINING COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nAdvantages of temporal model:")
    print("  ✅ Captures cardiac motion dynamics")
    print("  ✅ More realistic (uses full cardiac cycle)")
    print("  ✅ Potentially better clinical accuracy")
    print("\nGenerated files:")
    print("  - temporal_lstm.pt")
    print("  - temporal_lstm_metrics.csv")
    print("  - temporal_lstm_confusion_matrix.png")
    print("  - temporal_lstm_training_history.png")

if __name__ == "__main__":
    main()
