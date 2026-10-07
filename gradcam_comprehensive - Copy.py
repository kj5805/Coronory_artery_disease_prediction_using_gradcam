"""
Comprehensive Grad-CAM Analysis
Generates visualizations for correct and incorrect predictions
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd
import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt

class CADClassifier(nn.Module):
    def __init__(self, pretrained=False):
        super(CADClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 2)
        
    def forward(self, x):
        return self.resnet(x)

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.feature_maps = None
        self.gradients = None
        self.model.resnet.layer4.register_forward_hook(self.save_feature_maps)
        self.model.resnet.layer4.register_backward_hook(self.save_gradients)
    
    def save_feature_maps(self, module, input, output):
        self.feature_maps = output
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_image, target_class):
        output = self.model(input_image)
        self.model.zero_grad()
        
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam

def load_video_frame(video_path, frame_idx=0):
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
    
    cap.release()
    
    if frame is None:
        raise RuntimeError(f"Failed to read: {video_path}")
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def visualize_gradcam_detailed(model, image, true_label, pred_label, filename, save_dir, device):
    """Generate detailed Grad-CAM with prediction info"""
    
    gradcam = GradCAM(model)
    cam = gradcam.generate_cam(image, pred_label)
    cam_resized = cv2.resize(cam, (image.shape[3], image.shape[2]))
    
    img_np = image.squeeze().cpu().permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    overlay = 0.6 * img_np + 0.4 * heatmap
    overlay = overlay / overlay.max()
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_np)
    axes[0].set_title('Original Echo Image')
    axes[0].axis('off')
    
    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    # Add prediction info
    label_map = {0: 'Normal', 1: 'CAD-risk'}
    correct = "✓ CORRECT" if true_label == pred_label else "✗ INCORRECT"
    
    plt.suptitle(
        f'{filename}\n'
        f'True: {label_map[true_label]} | Predicted: {label_map[pred_label]} | {correct}',
        fontsize=12
    )
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'gradcam_{filename.replace(".avi", ".png")}')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Paths
    MODEL_PATH = "C:/Users/anany/dynamic/cad_output/best_cad_model.pt"
    VIDEO_DIR = "C:/Users/anany/dynamic/a4c-video-dir/Videos"
    TEST_CSV = "C:/Users/anany/dynamic/cad_output/test.csv"
    OUTPUT_DIR = "C:/Users/anany/dynamic/gradcam_analysis"
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = CADClassifier(pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    
    print("✅ Model loaded")
    
    # Transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load test data
    test_df = pd.read_csv(TEST_CSV)
    
    # Get predictions
    print("\nGenerating predictions...")
    predictions = []
    
    for idx in range(len(test_df)):
        row = test_df.iloc[idx]
        filename = row['FileName']
        true_label = row['CAD_Label']
        video_path = os.path.join(VIDEO_DIR, filename)
        
        try:
            frame = load_video_frame(video_path)
            frame_tensor = transform(frame).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(frame_tensor)
                pred_label = torch.argmax(output, dim=1).item()
            
            predictions.append({
                'filename': filename,
                'true_label': true_label,
                'pred_label': pred_label,
                'correct': true_label == pred_label
            })
        except:
            continue
    
    pred_df = pd.DataFrame(predictions)
    
    print(f"\nTotal predictions: {len(pred_df)}")
    print(f"Correct: {pred_df['correct'].sum()}")
    print(f"Incorrect: {(~pred_df['correct']).sum()}")
    
    # Categories for visualization
    categories = {
        'true_positives': pred_df[(pred_df['true_label']==1) & (pred_df['pred_label']==1)],
        'true_negatives': pred_df[(pred_df['true_label']==0) & (pred_df['pred_label']==0)],
        'false_positives': pred_df[(pred_df['true_label']==0) & (pred_df['pred_label']==1)],
        'false_negatives': pred_df[(pred_df['true_label']==1) & (pred_df['pred_label']==0)]
    }
    
    print("\n" + "="*60)
    print("GENERATING GRAD-CAM VISUALIZATIONS")
    print("="*60)
    
    # Generate Grad-CAMs for each category
    for category, df in categories.items():
        if len(df) == 0:
            print(f"\n{category}: No samples")
            continue
            
        cat_dir = os.path.join(OUTPUT_DIR, category)
        Path(cat_dir).mkdir(parents=True, exist_ok=True)
        
        # Take up to 5 samples from each category
        samples = df.sample(n=min(5, len(df)))
        
        print(f"\n{category}: Generating {len(samples)} visualizations...")
        
        for _, row in samples.iterrows():
            filename = row['filename']
            true_label = row['true_label']
            pred_label = row['pred_label']
            video_path = os.path.join(VIDEO_DIR, filename)
            
            try:
                frame = load_video_frame(video_path)
                frame_tensor = transform(frame).unsqueeze(0).to(device)
                
                visualize_gradcam_detailed(
                    model, frame_tensor, true_label, pred_label,
                    filename, cat_dir, device
                )
                print(f"  ✓ {filename}")
            except Exception as e:
                print(f"  ✗ {filename}: {e}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nGenerated categories:")
    print(f"  - true_positives/  (CAD-risk correctly identified)")
    print(f"  - true_negatives/  (Normal correctly identified)")
    print(f"  - false_positives/ (Normal misclassified as CAD-risk)")
    print(f"  - false_negatives/ (CAD-risk missed)")
    print("\nUse these in your presentation to show:")
    print("  ✅ What the model learned (TPs, TNs)")
    print("  ⚠️ Where it fails (FPs, FNs)")

if __name__ == "__main__":
    main()
