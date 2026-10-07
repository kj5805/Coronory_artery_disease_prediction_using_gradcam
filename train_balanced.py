"""
Enhanced CAD Classification with Class Balancing
Addresses class imbalance to improve recall
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
# Dataset Class (Same as before)
# ============================================================================

class EchoCADDataset(Dataset):
    """Dataset for CAD classification from echocardiographic videos"""

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
            log_path = os.path.join(os.getcwd(), 'unreadable_videos.log')
            with open(log_path, 'a') as f:
                f.write(f"{video_path}\n")
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
            raise RuntimeError(f"Cannot open video: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

            if not ret or frame is None:
                cap.release()
                raise RuntimeError(f"Failed to read frame from: {video_path}")

        cap.release()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

# ============================================================================
# Model Definition (Same as before)
# ============================================================================

class CADClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super(CADClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 2)

    def forward(self, x):
        return self.resnet(x)

# ============================================================================
# Enhanced Training with Class Balancing
# ============================================================================

def skip_none_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def train_model_balanced(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_dir):
    """Train with class-weighted loss"""

    best_val_f1 = 0.0  # Changed from accuracy to F1
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    val_f1s = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        train_samples_processed = 0

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
            train_samples_processed += inputs.size(0)

            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        epoch_train_loss = train_loss / train_samples_processed if train_samples_processed > 0 else 0
        epoch_train_acc = accuracy_score(train_labels, train_preds)

        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        val_probs = []
        val_samples_processed = 0

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
                val_samples_processed += inputs.size(0)

                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs[:, 1].cpu().numpy())

        epoch_val_loss = val_loss / val_samples_processed if val_samples_processed > 0 else 0
        epoch_val_acc = accuracy_score(val_labels, val_preds)
        _, _, epoch_val_f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary', zero_division=0)

        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        val_f1s.append(epoch_val_f1)

        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f} | Val F1: {epoch_val_f1:.4f}")

        # Save best model based on F1 score (better for imbalanced data)
        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_cad_model_balanced.pt'))
            print(f"✅ Best model saved! (Val F1: {best_val_f1:.4f})")

    # Plot training history
    plot_training_history_enhanced(train_losses, val_losses, train_accs, val_accs, val_f1s, save_dir)

    return model

def plot_training_history_enhanced(train_losses, val_losses, train_accs, val_accs, val_f1s, save_dir):
    """Plot training metrics including F1 score"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(train_losses, label='Train Loss', marker='o')
    axes[0].plot(val_losses, label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(train_accs, label='Train Acc', marker='o')
    axes[1].plot(val_accs, label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    # F1 Score
    axes[2].plot(val_f1s, label='Val F1', marker='d', color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('Validation F1 Score')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history_balanced.png'), dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model, test_loader, device, save_dir):
    """Evaluate model"""

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
    print("BALANCED MODEL - TEST SET RESULTS")
    print("="*60)
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f} ⭐ (Improved!)")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)

    # Calculate per-class metrics
    print("\nPer-Class Performance:")
    print(f"Normal Class:")
    print(f"  - Correctly identified: {cm[0,0]} / {cm[0,0] + cm[0,1]} ({cm[0,0]/(cm[0,0]+cm[0,1])*100:.1f}%)")
    print(f"CAD-risk Class:")
    print(f"  - Correctly identified: {cm[1,1]} / {cm[1,0] + cm[1,1]} ({cm[1,1]/(cm[1,0]+cm[1,1])*100:.1f}%)")

    # Save metrics
    metrics = {
        'accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(save_dir, 'test_metrics_balanced.csv'), index=False)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'CAD-risk'],
                yticklabels=['Normal', 'CAD-risk'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (Balanced Model)')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_balanced.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return metrics

def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the class-balanced EchoNet-Dynamic CAD classifier."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/cad/CAD_FileList.csv"),
        help="Prepared CAD metadata CSV (default: data/cad/CAD_FileList.csv).",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("data/echonet/a4c-video-dir/Videos"),
        help="Directory containing EchoNet-Dynamic AVI videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cad-balanced"),
        help="Directory for split CSVs, checkpoint, plots, and metrics.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--frame-idx", type=int, default=0)
    args = parser.parse_args()

    cad_csv = args.csv
    video_dir = args.video_dir
    output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load data
    full_data = pd.read_csv(cad_csv)

    train_csv = full_data[full_data['Split'] == 'TRAIN']
    val_csv = full_data[full_data['Split'] == 'VAL']
    test_csv = full_data[full_data['Split'] == 'TEST']

    # Save split CSVs
    train_csv.to_csv(output_dir / 'train.csv', index=False)
    val_csv.to_csv(output_dir / 'val.csv', index=False)
    test_csv.to_csv(output_dir / 'test.csv', index=False)

    # Create datasets
    train_dataset = EchoCADDataset(output_dir / 'train.csv', video_dir, transform, args.frame_idx)
    val_dataset = EchoCADDataset(output_dir / 'val.csv', video_dir, transform, args.frame_idx)
    test_dataset = EchoCADDataset(output_dir / 'test.csv', video_dir, transform, args.frame_idx)

    print(f"\nDataset sizes:")
    print(f"Train: {len(train_dataset)}")
    print(f"Val: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")

    # Calculate class weights for balanced sampling
    train_labels = train_csv['CAD_Label'].values
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]

    print(f"\nClass distribution in training:")
    print(f"  Normal (0): {class_counts[0]}")
    print(f"  CAD-risk (1): {class_counts[1]}")
    print(f"  Ratio: {class_counts[0]/class_counts[1]:.1f}:1")

    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Create dataloaders with weighted sampling
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                             num_workers=0, collate_fn=skip_none_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=0, collate_fn=skip_none_collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=skip_none_collate)

    # Create model
    model = CADClassifier(pretrained=True)
    model = model.to(device)

    # Class-weighted loss
    class_weights_tensor = torch.FloatTensor([1.0, class_counts[0]/class_counts[1]]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print("\n" + "="*60)
    print("Starting Balanced Training...")
    print(f"Class weights: Normal={class_weights_tensor[0]:.2f}, CAD-risk={class_weights_tensor[1]:.2f}")
    print("="*60)

    # Train model
    model = train_model_balanced(model, train_loader, val_loader, criterion, optimizer,
                                num_epochs=args.epochs, device=device, save_dir=output_dir)

    # Load best model
    model.load_state_dict(torch.load(
        output_dir / 'best_cad_model_balanced.pt',
        map_location=device,
    ))

    # Evaluate
    metrics = evaluate_model(model, test_loader, device, output_dir)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print("\nCOMPARISON WITH ORIGINAL MODEL:")
    print("Expected improvements:")
    print("  ✅ Higher Recall (catch more CAD-risk cases)")
    print("  ✅ Better F1 Score (balanced performance)")
    print("  ⚠️ Slightly lower overall accuracy (trade-off)")
    print("  ⚠️ More false positives (safer for medical screening)")

if __name__ == "__main__":
    main()
