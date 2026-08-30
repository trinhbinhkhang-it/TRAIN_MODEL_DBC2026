"""
Evaluation script for SwapFace Detector
Computes AUC, EER, Accuracy, Precision, Recall, F1, Confusion Matrix
Supports cross-dataset evaluation
"""

import os
import sys
import argparse
import yaml
import logging
import json
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_curve, auc, average_precision_score)
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from training.swapface_detector import SwapFaceDetector, create_swapface_detector
from training.train_swapface import FaceSwapVideoDataset, get_transforms


def compute_eer(y_true, y_score):
    """Compute Equal Error Rate (EER)"""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]
    return eer, eer_threshold


def evaluate_model(model, data_loader, device, logger, dataset_name='test'):
    """Evaluate model on a dataset"""
    model.eval()
    
    all_probs = []
    all_preds = []
    all_labels = []
    all_video_names = []
    all_frame_names = []
    
    inference_times = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {dataset_name}"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            video_names = batch['video_name']
            frame_names = batch['frame_name']
            
            # Time inference
            start_time = time.time()
            
            data_dict = {'image': images, 'label': labels}
            pred_dict = model(data_dict)
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            probs = pred_dict['prob'].cpu().numpy()
            preds = pred_dict['cls'].argmax(dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels_np)
            all_video_names.extend(video_names)
            all_frame_names.extend(frame_names)
    
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Frame-level metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0)
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except:
        roc_auc = 0.0
    
    try:
        ap = average_precision_score(all_labels, all_probs)
    except:
        ap = 0.0
    
    eer, eer_threshold = compute_eer(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Video-level metrics (average probabilities per video)
    video_probs = defaultdict(list)
    video_labels = {}
    for i, vname in enumerate(all_video_names):
        video_probs[vname].append(all_probs[i])
        video_labels[vname] = all_labels[i]
    
    video_probs_avg = np.array([np.mean(video_probs[v]) for v in video_probs])
    video_labels_arr = np.array([video_labels[v] for v in video_probs])
    video_preds = (video_probs_avg > 0.5).astype(int)
    
    video_acc = accuracy_score(video_labels_arr, video_preds)
    video_precision, video_recall, video_f1, _ = precision_recall_fscore_support(
        video_labels_arr, video_preds, average='binary', zero_division=0)
    
    try:
        video_auc = roc_auc_score(video_labels_arr, video_probs_avg)
    except:
        video_auc = 0.0
    
    try:
        video_ap = average_precision_score(video_labels_arr, video_probs_avg)
    except:
        video_ap = 0.0
    
    video_eer, video_eer_threshold = compute_eer(video_labels_arr, video_probs_avg)
    video_cm = confusion_matrix(video_labels_arr, video_preds)
    
    # Inference time stats
    avg_inference_time = np.mean(inference_times) * 1000  # ms
    p95_inference_time = np.percentile(inference_times, 95) * 1000  # ms
    fps = 1.0 / np.mean(inference_times) if np.mean(inference_times) > 0 else 0
    
    # Log results
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation Results: {dataset_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Frame-level Metrics:")
    logger.info(f"  Accuracy:  {acc:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1-Score:  {f1:.4f}")
    logger.info(f"  ROC-AUC:   {roc_auc:.4f}")
    logger.info(f"  AP:        {ap:.4f}")
    logger.info(f"  EER:       {eer:.4f} (threshold={eer_threshold:.4f})")
    logger.info(f"  Confusion Matrix:\n{cm}")
    logger.info(f"\nVideo-level Metrics:")
    logger.info(f"  Accuracy:  {video_acc:.4f}")
    logger.info(f"  Precision: {video_precision:.4f}")
    logger.info(f"  Recall:    {video_recall:.4f}")
    logger.info(f"  F1-Score:  {video_f1:.4f}")
    logger.info(f"  ROC-AUC:   {video_auc:.4f}")
    logger.info(f"  AP:        {video_ap:.4f}")
    logger.info(f"  EER:       {video_eer:.4f} (threshold={video_eer_threshold:.4f})")
    logger.info(f"  Confusion Matrix:\n{video_cm}")
    logger.info(f"\nInference Performance:")
    logger.info(f"  Avg Latency:  {avg_inference_time:.2f} ms")
    logger.info(f"  P95 Latency:  {p95_inference_time:.2f} ms")
    logger.info(f"  FPS:          {fps:.2f}")
    logger.info(f"{'='*60}\n")
    
    return {
        'frame_level': {
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'roc_auc': float(roc_auc),
            'average_precision': float(ap),
            'eer': float(eer),
            'eer_threshold': float(eer_threshold),
            'confusion_matrix': cm.tolist()
        },
        'video_level': {
            'accuracy': float(video_acc),
            'precision': float(video_precision),
            'recall': float(video_recall),
            'f1': float(video_f1),
            'roc_auc': float(video_auc),
            'average_precision': float(video_ap),
            'eer': float(video_eer),
            'eer_threshold': float(video_eer_threshold),
            'confusion_matrix': video_cm.tolist()
        },
        'inference': {
            'avg_latency_ms': float(avg_inference_time),
            'p95_latency_ms': float(p95_inference_time),
            'fps': float(fps)
        },
        'num_samples': len(all_labels),
        'num_videos': len(video_probs)
    }


def evaluate_thresholds(model, data_loader, device, logger, dataset_name='test'):
    """Evaluate at multiple thresholds"""
    model.eval()
    
    all_probs = []
    all_labels = []
    all_video_names = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Threshold eval {dataset_name}"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            video_names = batch['video_name']
            
            data_dict = {'image': images, 'label': labels}
            pred_dict = model(data_dict)
            
            probs = pred_dict['prob'].cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            all_probs.extend(probs)
            all_labels.extend(labels_np)
            all_video_names.extend(video_names)
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Video-level
    video_probs = defaultdict(list)
    video_labels = {}
    for i, vname in enumerate(all_video_names):
        video_probs[vname].append(all_probs[i])
        video_labels[vname] = all_labels[i]
    
    video_probs_avg = np.array([np.mean(video_probs[v]) for v in video_probs])
    video_labels_arr = np.array([video_labels[v] for v in video_probs])
    
    thresholds = np.arange(0.1, 1.0, 0.05)
    results = []
    
    for thresh in thresholds:
        preds = (video_probs_avg >= thresh).astype(int)
        acc = accuracy_score(video_labels_arr, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            video_labels_arr, preds, average='binary', zero_division=0)
        
        results.append({
            'threshold': float(thresh),
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1)
        })
    
    # Find best F1 threshold
    best_f1_idx = np.argmax([r['f1'] for r in results])
    best_threshold = results[best_f1_idx]['threshold']
    
    logger.info(f"\nThreshold Analysis ({dataset_name}):")
    logger.info(f"  Best F1 threshold: {best_threshold:.2f}")
    logger.info(f"  Best F1: {results[best_f1_idx]['f1']:.4f}")
    
    return results, best_threshold


def save_results(results, output_path, dataset_name):
    """Save evaluation results"""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # JSON
    json_file = output_path / f"evaluation_{dataset_name}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Markdown report
    md_file = output_path / f"evaluation_report_{dataset_name}.md"
    with open(md_file, 'w') as f:
        f.write(f"# Evaluation Report: {dataset_name}\n\n")
        
        # Frame-level
        f.write("## Frame-Level Metrics\n\n")
        frame = results['frame_level']
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Accuracy | {frame['accuracy']:.4f} |\n")
        f.write(f"| Precision | {frame['precision']:.4f} |\n")
        f.write(f"| Recall | {frame['recall']:.4f} |\n")
        f.write(f"| F1-Score | {frame['f1']:.4f} |\n")
        f.write(f"| ROC-AUC | {frame['roc_auc']:.4f} |\n")
        f.write(f"| Average Precision | {frame['average_precision']:.4f} |\n")
        f.write(f"| EER | {frame['eer']:.4f} |\n")
        f.write(f"| EER Threshold | {frame['eer_threshold']:.4f} |\n\n")
        
        # Confusion matrix
        f.write("### Confusion Matrix (Frame-Level)\n\n")
        cm = frame['confusion_matrix']
        f.write(f"| | Predicted Real | Predicted Fake |\n")
        f.write(f"|---|---|---|\n")
        f.write(f"| Actual Real | {cm[0][0]} | {cm[0][1]} |\n")
        f.write(f"| Actual Fake | {cm[1][0]} | {cm[1][1]} |\n\n")
        
        # Video-level
        f.write("## Video-Level Metrics\n\n")
        video = results['video_level']
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Accuracy | {video['accuracy']:.4f} |\n")
        f.write(f"| Precision | {video['precision']:.4f} |\n")
        f.write(f"| Recall | {video['recall']:.4f} |\n")
        f.write(f"| F1-Score | {video['f1']:.4f} |\n")
        f.write(f"| ROC-AUC | {video['roc_auc']:.4f} |\n")
        f.write(f"| Average Precision | {video['average_precision']:.4f} |\n")
        f.write(f"| EER | {video['eer']:.4f} |\n")
        f.write(f"| EER Threshold | {video['eer_threshold']:.4f} |\n\n")
        
        # Confusion matrix
        f.write("### Confusion Matrix (Video-Level)\n\n")
        cm = video['confusion_matrix']
        f.write(f"| | Predicted Real | Predicted Fake |\n")
        f.write(f"|---|---|---|\n")
        f.write(f"| Actual Real | {cm[0][0]} | {cm[0][1]} |\n")
        f.write(f"| Actual Fake | {cm[1][0]} | {cm[1][1]} |\n\n")
        
        # Inference
        f.write("## Inference Performance\n\n")
        inf = results['inference']
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Avg Latency | {inf['avg_latency_ms']:.2f} ms |\n")
        f.write(f"| P95 Latency | {inf['p95_latency_ms']:.2f} ms |\n")
        f.write(f"| FPS | {inf['fps']:.2f} |\n\n")
        
        f.write(f"---\n*Samples: {results['num_samples']}, Videos: {results['num_videos']}*\n")
    
    logger.info(f"Results saved to {json_file} and {md_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate SwapFace Detector')
    parser.add_argument('--config', type=str, default='configs/swapface_detector.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--data_root', type=str, default='data/',
                        help='Root directory for dataset')
    parser.add_argument('--split', type=str, default='test',
                        help='Split to evaluate (test/val)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='reports/',
                        help='Output directory for results')
    parser.add_argument('--threshold_analysis', action='store_true',
                        help='Run threshold analysis')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model_config = config.get('model', {})
    model = create_swapface_detector(model_config).to(device)
    
    checkpoint = torch.load(args.model, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    logger.info(f"Loaded model from {args.model}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / (1024 * 1024)  # FP32
    logger.info(f"Model parameters: {total_params:,} ({model_size_mb:.2f} MB FP32)")
    
    # Data loader
    val_transform = get_transforms(config, is_train=False)
    
    test_dataset = FaceSwapVideoDataset(args.data_root, args.split, config, val_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate
    results = evaluate_model(model, test_loader, device, logger, args.split)
    
    # Add model info
    results['model_info'] = {
        'parameters': total_params,
        'size_mb_fp32': model_size_mb,
        'checkpoint': args.model
    }
    
    # Threshold analysis
    if args.threshold_analysis:
        threshold_results, best_thresh = evaluate_thresholds(
            model, test_loader, device, logger, args.split)
        results['threshold_analysis'] = threshold_results
        results['best_threshold'] = best_thresh
    
    # Save results
    save_results(results, args.output_dir, args.split)
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()