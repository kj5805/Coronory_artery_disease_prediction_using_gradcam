"""
CAD Risk Classification Model with Grad-CAM Explainability
Uses 2D ResNet architecture on echocardiographic frames

Author: Ananya Agrawal
Project: AI-Based Prediction of CAD from Echocardiographic Images
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
    """
    Dataset for CAD classification from echocardiographic videos
    Extracts single frame from each video
    """
    
    def __init__(self, csv_file, video_dir, transform=None, frame_idx=0):
        """
        Args:
            csv_file: Path to CAD_FileList.csv
            video_dir: Path to Videos directory
            transform: Optional transform to apply
            frame_idx: Which frame to extract (0 = first frame)
        """
        self.data = pd.read_csv(csv_file)
        self.video_dir = video_dir
        self.transform = transform
        self.frame_idx = frame_idx
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get video filename and label
        row = self.data.iloc[idx]
        video_name = row['FileName']
        label = row['CAD_Label']
        
        # Load video and extract frame
        video_path = os.path.join(self.video_dir, video_name)
        try:
            frame = self.load_video_frame(video_path, self.frame_idx)
        except Exception as e:
            # Log the unreadable video
            log_path = os.path.join(os.getcwd(), 'unreadable_videos.log')
            with open(log_path, 'a') as f:
                f.write(f"{video_path}\n")
            # Return None to be filtered out by collate_fn
            return None
        
        if self.transform:
            frame = self.transform(frame)
        
        return frame, label, video_name
    
    def load_video_frame(self, video_path, frame_idx):
        """Extract a single frame from video"""
        # Add .avi extension if not present
        if not video_path.endswith('.avi'):
            video_path = video_path + '.avi'
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret or frame is None:
            # If frame_idx is out of range, get first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        
        cap.release()
        
        # Check if frame is valid before color conversion
        if frame is None or not ret:
            raise RuntimeError(f"Failed to read frame from: {video_path}")
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame

# ============================================================================
# Model Definition
# ============================================================================

class CADClassifier(nn.Module):
    """
    Binary classifier for CAD risk prediction
    Uses pretrained ResNet-18 backbone
    """
    
    def __init__(self, pretrained=True):
        super(CADClassifier, self).__init__()
        
        # Load pretrained ResNet-18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Modify final layer for binary classification
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 2)  # 2 classes: Normal vs CAD-risk
        
    def forward(self, x):
        return self.resnet(x)
    
    def get_feature_maps(self, x):
        """Get feature maps from last conv layer (for Grad-CAM)"""
        # Forward through all layers except fc
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)  # This is our feature map
        
        return x

# ============================================================================
# Grad-CAM Implementation
# ============================================================================

class GradCAM:
    """
    Grad-CAM for visualizing what the model focuses on
    """
    
    def __init__(self, model):
        self.model = model
        self.feature_maps = None
        self.gradients = None
        
        # Register hooks
        self.model.resnet.layer4.register_forward_hook(self.save_feature_maps)
        self.model.resnet.layer4.register_backward_hook(self.save_gradients)
    
    def save_feature_maps(self, module, input, output):
        self.feature_maps = output
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_image, target_class):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_image: Input tensor [1, 3, H, W]
            target_class: Target class index (0 or 1)
        
        Returns:
            cam: Heatmap as numpy array
        """
        # Forward pass
        output = self.model(input_image)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get weights (global average pooling of gradients)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # Weighted combination of feature maps
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        cam = torch.relu(cam)  # ReLU
        
        # Normalize
        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam

# ============================================================================
# Training Function
# ============================================================================

def skip_none_collate(batch):
    """Custom collate function to skip None samples"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_dir):
    """
    Train the CAD classification model
    """
    
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
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
            if batch is None:  # Skip if entire batch failed to load
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
                if batch is None:  # Skip if entire batch failed to load
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
                val_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1
        
        epoch_val_loss = val_loss / val_samples_processed if val_samples_processed > 0 else 0
        epoch_val_acc = accuracy_score(val_labels, val_preds)
        
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_cad_model.pt'))
            print(f"✅ Best model saved! (Val Acc: {best_val_acc:.4f})")
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs, save_dir)
    
    return model

# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_model(model, test_loader, device, save_dir):
    """
    Evaluate model on test set and generate metrics
    """
    
    model.eval()
    test_preds = []
    test_labels = []
    test_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:  # Skip if entire batch failed to load
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
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='binary')
    auc = roc_auc_score(test_labels, test_probs)
    cm = confusion_matrix(test_labels, test_preds)
    
    print("\n" + "="*60)
    print("TEST SET RESULTS")
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
        'accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc
    }
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(save_dir, 'test_metrics.csv'), index=False)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, save_dir)
    
    return metrics

# ============================================================================
# Visualization Functions
# ============================================================================

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_dir):
    """Plot training and validation metrics"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Acc', marker='o')
    ax2.plot(val_accs, label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, save_dir):
    """Plot confusion matrix"""
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'CAD-risk'],
                yticklabels=['Normal', 'CAD-risk'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def visualize_gradcam(model, image, label, filename, save_dir, device):
    """Generate and save Grad-CAM visualization"""
    
    gradcam = GradCAM(model)
    
    # Generate CAM
    cam = gradcam.generate_cam(image, label)
    
    # Resize CAM to image size
    cam_resized = cv2.resize(cam, (image.shape[3], image.shape[2]))
    
    # Convert image tensor to numpy
    img_np = image.squeeze().cpu().permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap / 255.0
    
    # Overlay
    overlay = 0.6 * img_np + 0.4 * heatmap
    overlay = overlay / overlay.max()
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.suptitle(f'Grad-CAM Visualization - {filename}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'gradcam_{filename.replace(".avi", ".png")}'), 
                dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# Main Training Script
# ============================================================================

def main():
    # Paths (update according to your setup)
    DATA_DIR = "C:/Users/anany/dynamic"
    CAD_CSV = "C:/Users/anany/dynamic/cad_data/CAD_FileList.csv"
    VIDEO_DIR = "C:/Users/anany/dynamic/a4c-video-dir/Videos"  # Correct path
    OUTPUT_DIR = "C:/Users/anany/dynamic/cad_output"
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Device
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
    full_data = pd.read_csv(CAD_CSV)
    
    train_csv = full_data[full_data['Split'] == 'TRAIN']
    val_csv = full_data[full_data['Split'] == 'VAL']
    test_csv = full_data[full_data['Split'] == 'TEST']
    
    # Save split CSVs
    train_csv.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
    val_csv.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'), index=False)
    test_csv.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)
    
    # Create datasets
    train_dataset = EchoCADDataset(os.path.join(OUTPUT_DIR, 'train.csv'), VIDEO_DIR, transform)
    val_dataset = EchoCADDataset(os.path.join(OUTPUT_DIR, 'val.csv'), VIDEO_DIR, transform)
    test_dataset = EchoCADDataset(os.path.join(OUTPUT_DIR, 'test.csv'), VIDEO_DIR, transform)
    
    # Create dataloaders with custom collate_fn to skip None
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, collate_fn=skip_none_collate)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=skip_none_collate)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=skip_none_collate)
    
    print(f"\nDataset sizes:")
    print(f"Train: {len(train_dataset)}")
    print(f"Val: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")
    
    # Create model
    model = CADClassifier(pretrained=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Train model
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)
    
    model = train_model(model, train_loader, val_loader, criterion, optimizer, 
                       num_epochs=10, device=device, save_dir=OUTPUT_DIR)
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_cad_model.pt')))
    
    # Evaluate
    metrics = evaluate_model(model, test_loader, device, OUTPUT_DIR)
    
    # Generate Grad-CAM for sample test images
    print("\n" + "="*60)
    print("Generating Grad-CAM Visualizations...")
    print("="*60)
    
    model.eval()
    gradcam_dir = os.path.join(OUTPUT_DIR, 'gradcam_samples')
    Path(gradcam_dir).mkdir(parents=True, exist_ok=True)
    
    # Get 5 samples from test set
    sample_indices = np.random.choice(len(test_dataset), min(5, len(test_dataset)), replace=False)
    
    for idx in sample_indices:
        image, label, filename = test_dataset[idx]
        image_batch = image.unsqueeze(0).to(device)
        
        visualize_gradcam(model, image_batch, label, filename, gradcam_dir, device)
        print(f"Generated Grad-CAM for {filename}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - best_cad_model.pt")
    print(f"  - training_history.png")
    print(f"  - confusion_matrix.png")
    print(f"  - test_metrics.csv")
    print(f"  - gradcam_samples/")

if __name__ == "__main__":
    main()