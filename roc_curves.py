"""
ROC Curve and Advanced Performance Visualizations
Generates publication-quality performance analysis plots
"""

import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd
import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, classification_report,
    roc_auc_score
)
import seaborn as sns

class CADClassifier(nn.Module):
    def __init__(self, pretrained=False):
        super(CADClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 2)

    def forward(self, x):
        return self.resnet(x)

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

def generate_roc_curve(y_true, y_probs, save_path):
    """Generate ROC curve"""

    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))

    # Plot ROC curve
    plt.plot(fpr, tpr, color='#2E86AB', lw=3,
             label=f'CAD Classifier (AUC = {roc_auc:.3f})')

    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--',
             label='Random Classifier (AUC = 0.500)')

    # Optimal threshold point (Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]

    plt.plot(optimal_fpr, optimal_tpr, 'ro', markersize=12,
             label=f'Optimal Threshold = {optimal_threshold:.3f}')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curve',
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3)

    # Add text box with key metrics
    textstr = f'AUC = {roc_auc:.3f}\nSensitivity at optimal = {optimal_tpr:.3f}\nSpecificity at optimal = {1-optimal_fpr:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.6, 0.2, textstr, fontsize=11, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ ROC curve saved to: {save_path}")
    print(f"   AUC-ROC: {roc_auc:.4f}")
    print(f"   Optimal threshold: {optimal_threshold:.4f}")
    print(f"   Sensitivity at optimal: {optimal_tpr:.4f}")
    print(f"   Specificity at optimal: {1-optimal_fpr:.4f}")

    return roc_auc, optimal_threshold

def generate_precision_recall_curve(y_true, y_probs, save_path):
    """Generate Precision-Recall curve"""

    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)

    plt.figure(figsize=(10, 8))

    plt.plot(recall, precision, color='#A23B72', lw=3,
             label=f'CAD Classifier (AP = {avg_precision:.3f})')

    # Baseline (random classifier for imbalanced data)
    baseline = np.sum(y_true) / len(y_true)
    plt.plot([0, 1], [baseline, baseline], color='gray', lw=2,
             linestyle='--', label=f'Random Classifier (AP = {baseline:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=14, fontweight='bold')
    plt.ylabel('Precision', fontsize=14, fontweight='bold')
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="upper right", fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3)

    # Add text box
    textstr = f'Average Precision = {avg_precision:.3f}\nBaseline = {baseline:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.05, 0.15, textstr, fontsize=11, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Precision-Recall curve saved to: {save_path}")
    print(f"   Average Precision: {avg_precision:.4f}")

    return avg_precision

def generate_threshold_analysis(y_true, y_probs, save_path):
    """Analyze performance at different thresholds"""

    thresholds = np.linspace(0, 1, 100)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)

        # Calculate metrics
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: All metrics
    ax1.plot(thresholds, accuracies, label='Accuracy', linewidth=2.5, color='#2E86AB')
    ax1.plot(thresholds, precisions, label='Precision', linewidth=2.5, color='#A23B72')
    ax1.plot(thresholds, recalls, label='Recall', linewidth=2.5, color='#F18F01')
    ax1.plot(thresholds, f1_scores, label='F1-Score', linewidth=2.5, color='#06A77D')

    # Mark default threshold (0.5)
    default_idx = 50
    ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Default (0.5)')

    # Mark optimal F1 threshold
    optimal_f1_idx = np.argmax(f1_scores)
    optimal_f1_threshold = thresholds[optimal_f1_idx]
    ax1.axvline(x=optimal_f1_threshold, color='green', linestyle='--',
                linewidth=2, alpha=0.5, label=f'Optimal F1 ({optimal_f1_threshold:.2f})')

    ax1.set_xlabel('Classification Threshold', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax1.set_title('Performance Metrics vs Threshold', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Right plot: Zoom on optimal region
    zoom_start = max(0, optimal_f1_threshold - 0.2)
    zoom_end = min(1, optimal_f1_threshold + 0.2)

    mask = (thresholds >= zoom_start) & (thresholds <= zoom_end)

    ax2.plot(thresholds[mask], accuracies[mask] if isinstance(accuracies, np.ndarray) else np.array(accuracies)[mask],
             label='Accuracy', linewidth=2.5, color='#2E86AB')
    ax2.plot(thresholds[mask], precisions[mask] if isinstance(precisions, np.ndarray) else np.array(precisions)[mask],
             label='Precision', linewidth=2.5, color='#A23B72')
    ax2.plot(thresholds[mask], recalls[mask] if isinstance(recalls, np.ndarray) else np.array(recalls)[mask],
             label='Recall', linewidth=2.5, color='#F18F01')
    ax2.plot(thresholds[mask], f1_scores[mask] if isinstance(f1_scores, np.ndarray) else np.array(f1_scores)[mask],
             label='F1-Score', linewidth=2.5, color='#06A77D')

    ax2.axvline(x=optimal_f1_threshold, color='green', linestyle='--',
                linewidth=2, alpha=0.7)

    ax2.set_xlabel('Classification Threshold', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax2.set_title(f'Zoomed View (Optimal Region)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Threshold analysis saved to: {save_path}")
    print(f"   Optimal F1 threshold: {optimal_f1_threshold:.4f}")
    print(f"   F1 score at optimal: {f1_scores[optimal_f1_idx]:.4f}")

def generate_performance_summary(y_true, y_probs, save_path):
    """Generate comprehensive performance summary"""

    # Use default threshold 0.5
    y_pred = (y_probs >= 0.5).astype(int)

    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=['Normal', 'CAD-risk'],
                                   output_dict=True)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Per-class metrics
    ax1 = axes[0, 0]
    classes = ['Normal', 'CAD-risk']
    metrics = ['Precision', 'Recall', 'F1-Score']

    x = np.arange(len(classes))
    width = 0.25

    precisions = [report['Normal']['precision'], report['CAD-risk']['precision']]
    recalls = [report['Normal']['recall'], report['CAD-risk']['recall']]
    f1s = [report['Normal']['f1-score'], report['CAD-risk']['f1-score']]

    ax1.bar(x - width, precisions, width, label='Precision', color='#2E86AB', alpha=0.8)
    ax1.bar(x, recalls, width, label='Recall', color='#A23B72', alpha=0.8)
    ax1.bar(x + width, f1s, width, label='F1-Score', color='#06A77D', alpha=0.8)

    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Per-Class Performance Metrics', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)

    # 2. Support (sample counts)
    ax2 = axes[0, 1]
    supports = [report['Normal']['support'], report['CAD-risk']['support']]
    colors_support = ['#4CAF50', '#FF9800']
    bars = ax2.bar(classes, supports, color=colors_support, alpha=0.7, edgecolor='black')

    for bar, support in zip(bars, supports):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(support)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax2.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax2.set_title('Class Distribution in Test Set', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # 3. Overall metrics
    ax3 = axes[1, 0]
    overall_metrics = {
        'Accuracy': report['accuracy'],
        'Macro Avg\nPrecision': report['macro avg']['precision'],
        'Macro Avg\nRecall': report['macro avg']['recall'],
        'Macro Avg\nF1': report['macro avg']['f1-score'],
        'Weighted Avg\nF1': report['weighted avg']['f1-score']
    }

    metric_names = list(overall_metrics.keys())
    metric_values = list(overall_metrics.values())
    colors_overall = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#9C27B0']

    bars = ax3.barh(metric_names, metric_values, color=colors_overall, alpha=0.7, edgecolor='black')

    for bar, value in zip(bars, metric_values):
        width = bar.get_width()
        ax3.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                f'{value:.3f}',
                ha='left', va='center', fontsize=11, fontweight='bold')

    ax3.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax3.set_title('Overall Performance Metrics', fontsize=13, fontweight='bold')
    ax3.set_xlim([0, 1.1])
    ax3.grid(axis='x', alpha=0.3)

    # 4. Key statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')

    auc_score = roc_auc_score(y_true, y_probs)

    stats_data = [
        ['Total Samples', f'{len(y_true)}'],
        ['Positive Class (CAD-risk)', f'{int(report["CAD-risk"]["support"])} ({report["CAD-risk"]["support"]/len(y_true)*100:.1f}%)'],
        ['Negative Class (Normal)', f'{int(report["Normal"]["support"])} ({report["Normal"]["support"]/len(y_true)*100:.1f}%)'],
        ['', ''],
        ['Overall Accuracy', f'{report["accuracy"]:.4f}'],
        ['AUC-ROC', f'{auc_score:.4f}'],
        ['', ''],
        ['CAD-risk Precision', f'{report["CAD-risk"]["precision"]:.4f}'],
        ['CAD-risk Recall', f'{report["CAD-risk"]["recall"]:.4f}'],
        ['CAD-risk F1-Score', f'{report["CAD-risk"]["f1-score"]:.4f}'],
    ]

    table = ax4.table(cellText=stats_data, cellLoc='left',
                     colWidths=[0.6, 0.4],
                     loc='center',
                     bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style the table
    for i in range(len(stats_data)):
        cell = table[(i, 0)]
        cell.set_facecolor('#E8F4F8' if i % 2 == 0 else 'white')
        cell.set_text_props(weight='bold')

        cell = table[(i, 1)]
        cell.set_facecolor('#E8F4F8' if i % 2 == 0 else 'white')

    ax4.set_title('Performance Summary', fontsize=13, fontweight='bold', pad=20)

    plt.suptitle('Comprehensive Model Performance Analysis',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Performance summary saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate ROC and performance plots.")
    parser.add_argument("--model", type=Path, default=Path("outputs/cad-balanced/best_cad_model_balanced.pt"))
    parser.add_argument("--video-dir", type=Path, default=Path("data/echonet/a4c-video-dir/Videos"))
    parser.add_argument("--test-csv", type=Path, default=Path("outputs/cad-balanced/test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/performance"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = CADClassifier(pretrained=False)
    model.load_state_dict(torch.load(args.model, map_location=device))
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

    # Load test data and generate predictions
    print("\nGenerating predictions on test set...")
    test_df = pd.read_csv(args.test_csv)

    y_true = []
    y_probs = []

    for idx in range(len(test_df)):
        row = test_df.iloc[idx]
        filename = row['FileName']
        true_label = row['CAD_Label']
        video_path = os.path.join(args.video_dir, filename)

        try:
            frame = load_video_frame(video_path)
            frame_tensor = transform(frame).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(frame_tensor)
                prob = torch.softmax(output, dim=1)[0, 1].item()  # Probability of class 1

            y_true.append(true_label)
            y_probs.append(prob)
        except:
            continue

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(test_df)} samples...")

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    print(f"\n✅ Predictions complete: {len(y_true)} samples")

    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING PERFORMANCE VISUALIZATIONS")
    print("="*60)

    # 1. ROC Curve
    print("\n1. ROC Curve...")
    generate_roc_curve(y_true, y_probs,
                      os.path.join(args.output_dir, 'roc_curve.png'))

    # 2. Precision-Recall Curve
    print("\n2. Precision-Recall Curve...")
    generate_precision_recall_curve(y_true, y_probs,
                                   os.path.join(args.output_dir, 'precision_recall_curve.png'))

    # 3. Threshold Analysis
    print("\n3. Threshold Analysis...")
    generate_threshold_analysis(y_true, y_probs,
                               os.path.join(args.output_dir, 'threshold_analysis.png'))

    # 4. Performance Summary
    print("\n4. Performance Summary...")
    generate_performance_summary(y_true, y_probs,
                                os.path.join(args.output_dir, 'performance_summary.png'))

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nAll visualizations saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("  ✅ roc_curve.png")
    print("  ✅ precision_recall_curve.png")
    print("  ✅ threshold_analysis.png")
    print("  ✅ performance_summary.png")
    print("\nUse these in your presentation/report to show:")
    print("  📊 Discrimination ability (ROC)")
    print("  📊 Precision-recall trade-off")
    print("  📊 Optimal threshold selection")
    print("  📊 Comprehensive performance metrics")

if __name__ == "__main__":
    main()
