"""
Training script for SwapFace Detector
Supports EfficientNet-B0 with DeepfakeBench integration
"""

import os
import sys
import argparse
import yaml
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from training.swapface_detector import SwapFaceDetector, create_swapface_detector
from training.efficientnet_b0_backbone import create_efficientnet_b0


class FaceSwapVideoDataset(Dataset):
    """
    Dataset for face swap detection
    Expects pre-extracted face frames organized as:
    data/
        train/
            real/
                video1_frame0.jpg
                video1_frame1.jpg
                ...
            fake/
                video1_frame0.jpg
                ...
        val/
            real/
            fake/
        test/
            real/
            fake/
    Or JSON metadata format
    """
    
    def __init__(self, data_root: str, split: str, config: dict, transform=None):
        self.data_root = Path(data_root)
        self.split = split
        self.config = config
        self.transform = transform
        
        self.input_size = config.get('input_size', [224, 224])
        self.frame_per_video = config.get('frame_per_video', 8)
        
        # Load data
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split}")
    
    def _load_samples(self):
        """Load samples from directory or JSON"""
        samples = []
        
        split_dir = self.data_root / self.split
        if not split_dir.exists():
            # Try JSON format
            json_file = self.data_root / f"{self.split}.json"
            if json_file.exists():
                return self._load_from_json(json_file)
            else:
                raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        # Directory format: split/real/ and split/fake/
        for label, label_name in enumerate(['real', 'fake']):
            label_dir = split_dir / label_name
            if not label_dir.exists():
                continue
            
            for video_dir in label_dir.iterdir():
                if not video_dir.is_dir():
                    continue
                
                frames = sorted(list(video_dir.glob('*.jpg')) + list(video_dir.glob('*.png')))
                if len(frames) == 0:
                    continue
                
                # Sample frames
                if len(frames) > self.frame_per_video:
                    indices = np.linspace(0, len(frames) - 1, self.frame_per_video, dtype=int)
                    frames = [frames[i] for i in indices]
                
                for frame_path in frames:
                    samples.append({
                        'path': str(frame_path),
                        'label': label,
                        'video_name': video_dir.name,
                        'frame_name': frame_path.name
                    })
        
        return samples
    
    def _load_from_json(self, json_file: Path):
        """Load from JSON metadata"""
        import json
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            samples.append({
                'path': item['path'],
                'label': item['label'],
                'video_name': item.get('video_name', 'unknown'),
                'frame_name': Path(item['path']).name
            })
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        import cv2
        img = cv2.imread(sample['path'])
        if img is None:
            # Return dummy if failed
            img = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.input_size[1], self.input_size[0]))
        
        # Apply transforms
        if self.transform:
            img = self.transform(image=img)['image']
            # ToTensorV2 already returns tensor
            if not isinstance(img, torch.Tensor):
                img = torch.from_numpy(img).float()
        else:
            # Default normalization
            img = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = (img - mean) / std
            img = img.transpose(2, 0, 1)  # HWC to CHW
            img = torch.from_numpy(img).float()
        
        return {
            'image': img,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'video_name': sample['video_name'],
            'frame_name': sample['frame_name']
        }


def get_transforms(config: dict, is_train: bool = True):
    """Get albumentations transforms"""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    input_size = config.get('input_size', [224, 224])
    
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ImageCompression(quality_lower=40, quality_upper=100, p=0.3),
            A.Resize(input_size[0], input_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(input_size[0], input_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


class Trainer:
    """Trainer for SwapFace Detector"""
    
    def __init__(self, config: dict, model: SwapFaceDetector, 
                 optimizer: optim.Optimizer, scheduler, 
                 logger: logging.Logger, device: torch.device):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.device = device
        
        # Setup logging
        self.writer = SummaryWriter(config.get('log_dir', 'logs/tensorboard'))
        self.best_metric = 0.0
        self.best_epoch = 0
        
        # Metrics
        self.metric_scoring = config.get('metric_scoring', 'auc')
        
    def train_epoch(self, epoch: int, train_loader: DataLoader, 
                    val_loader: DataLoader = None) -> float:
        """Train one epoch"""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward
            data_dict = {'image': images, 'label': labels}
            pred_dict = self.model(data_dict)
            
            # Loss
            loss_dict = self.model.get_losses(data_dict, pred_dict)
            loss = loss_dict['overall']
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                               self.config['gradient_clip'])
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            probs = pred_dict['prob'].detach().cpu().numpy()
            preds = pred_dict['cls'].argmax(dim=1).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels_np)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log to tensorboard
            global_step = epoch * len(train_loader) + batch_idx
            self.writer.add_scalar('train/loss', loss.item(), global_step)
        
        # Compute epoch metrics
        avg_loss = total_loss / len(train_loader)
        acc = accuracy_score(all_labels, all_preds)
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0)
        
        self.logger.info(f"Epoch {epoch} Train: Loss={avg_loss:.4f}, Acc={acc:.4f}, AUC={auc:.4f}, F1={f1:.4f}")
        self.writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        self.writer.add_scalar('train/epoch_acc', acc, epoch)
        self.writer.add_scalar('train/epoch_auc', auc, epoch)
        self.writer.add_scalar('train/epoch_f1', f1, epoch)
        
        # Validation
        val_metric = 0.0
        if val_loader is not None:
            val_metric = self.validate(epoch, val_loader)
        
        # Scheduler step
        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metric)
            else:
                self.scheduler.step()
        
        return val_metric
    
    def validate(self, epoch: int, val_loader: DataLoader) -> float:
        """Validate model"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        video_preds = defaultdict(list)
        video_labels = {}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                video_names = batch['video_name']
                
                data_dict = {'image': images, 'label': labels}
                pred_dict = self.model(data_dict)
                
                probs = pred_dict['prob'].cpu().numpy()
                preds = pred_dict['cls'].argmax(dim=1).cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels_np)
                
                # Video-level aggregation
                for i, vname in enumerate(video_names):
                    video_preds[vname].append(probs[i])
                    video_labels[vname] = labels_np[i]
        
        # Frame-level metrics
        acc = accuracy_score(all_labels, all_preds)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        
        # Video-level metrics (average frame probs per video)
        video_probs = [np.mean(video_preds[v]) for v in video_preds]
        video_labels_list = [video_labels[v] for v in video_preds]
        video_preds_list = [1 if p > 0.5 else 0 for p in video_probs]
        
        video_acc = accuracy_score(video_labels_list, video_preds_list)
        try:
            video_auc = roc_auc_score(video_labels_list, video_probs)
        except:
            video_auc = 0.0
        
        self.logger.info(f"Epoch {epoch} Val (Frame): Acc={acc:.4f}, AUC={auc:.4f}, F1={f1:.4f}")
        self.logger.info(f"Epoch {epoch} Val (Video): Acc={video_acc:.4f}, AUC={video_auc:.4f}")
        self.logger.info(f"Confusion Matrix:\n{cm}")
        
        self.writer.add_scalar('val/frame_acc', acc, epoch)
        self.writer.add_scalar('val/frame_auc', auc, epoch)
        self.writer.add_scalar('val/frame_f1', f1, epoch)
        self.writer.add_scalar('val/video_acc', video_acc, epoch)
        self.writer.add_scalar('val/video_auc', video_auc, epoch)
        
        # Return primary metric for model selection
        if self.metric_scoring == 'auc':
            return auc
        elif self.metric_scoring == 'acc':
            return acc
        elif self.metric_scoring == 'f1':
            return f1
        else:
            return auc
    
    def save_checkpoint(self, epoch: int, metric: float, is_best: bool, save_dir: str):
        """Save model checkpoint"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metric': metric,
            'config': self.config
        }
        
        # Save latest
        latest_path = save_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = save_dir / 'best_swapface_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model with {self.metric_scoring}={metric:.4f} to {best_path}")


def create_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    """Create optimizer from config"""
    opt_type = config.get('optimizer', {}).get('type', 'adamw')
    lr = config.get('learning_rate', 1e-4)
    weight_decay = config.get('weight_decay', 1e-5)
    
    if opt_type == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == 'adam':
        opt_config = config.get('optimizer', {}).get('adam', {})
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay,
                         betas=(opt_config.get('beta1', 0.9), opt_config.get('beta2', 0.999)),
                         eps=opt_config.get('eps', 1e-8))
    elif opt_type == 'sgd':
        opt_config = config.get('optimizer', {}).get('sgd', {})
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                        momentum=opt_config.get('momentum', 0.9))
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")


def create_scheduler(optimizer: optim.Optimizer, config: dict):
    """Create learning rate scheduler"""
    scheduler_type = config.get('scheduler', 'cosine')
    epochs = config.get('epochs', 50)
    
    if scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == 'step':
        step_size = config.get('lr_step', 10)
        gamma = config.get('lr_gamma', 0.5)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                     factor=0.5, patience=5)
    elif scheduler_type == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    return logger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train SwapFace Detector')
    parser.add_argument('--config', type=str, default='configs/swapface_detector.yaml',
                        help='Path to config file')
    parser.add_argument('--data_root', type=str, default='data/',
                        help='Root directory for dataset')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with command line args
    if args.data_root:
        config['data_root'] = args.data_root
    
    # Setup
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup logging
    log_dir = config.get('paths', {}).get('logs', 'logs/')
    logger = setup_logging(log_dir)
    logger.info(f"Config: {config}")
    
    # Create model
    model_config = config.get('model', {})
    model = create_swapface_detector(model_config).to(device)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create data loaders
    train_transform = get_transforms(config, is_train=True)
    val_transform = get_transforms(config, is_train=False)
    
    data_root = config.get('data_root', 'data/')
    
    train_dataset = FaceSwapVideoDataset(data_root, 'train', config, train_transform)
    val_dataset = FaceSwapVideoDataset(data_root, 'val', config, val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('training', {}).get('batch_size', 32),
        shuffle=True,
        num_workers=config.get('training', {}).get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('training', {}).get('batch_size', 32),
        shuffle=False,
        num_workers=config.get('training', {}).get('num_workers', 4),
        pin_memory=True
    )
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config.get('training', {}))
    scheduler = create_scheduler(optimizer, config.get('training', {}))
    
    # Resume if needed
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Create trainer
    trainer = Trainer(config, model, optimizer, scheduler, logger, device)
    
    # Training loop
    epochs = config.get('training', {}).get('epochs', 50)
    save_dir = config.get('paths', {}).get('checkpoints', 'checkpoints/')
    
    logger.info(f"Starting training for {epochs} epochs")
    
    for epoch in range(start_epoch, epochs + 1):
        val_metric = trainer.train_epoch(epoch, train_loader, val_loader)
        
        # Save checkpoint
        is_best = val_metric > trainer.best_metric
        if is_best:
            trainer.best_metric = val_metric
            trainer.best_epoch = epoch
        
        trainer.save_checkpoint(epoch, val_metric, is_best, save_dir)
        
        logger.info(f"Best {config.get('metric_scoring', 'auc')}: {trainer.best_metric:.4f} at epoch {trainer.best_epoch}")
    
    logger.info("Training completed!")
    logger.info(f"Best metric: {trainer.best_metric:.4f} at epoch {trainer.best_epoch}")
    
    # Save final model
    final_path = Path(save_dir) / 'final_swapface_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_metric': trainer.best_metric,
        'best_epoch': trainer.best_epoch
    }, final_path)
    logger.info(f"Final model saved to {final_path}")


if __name__ == "__main__":
    main()