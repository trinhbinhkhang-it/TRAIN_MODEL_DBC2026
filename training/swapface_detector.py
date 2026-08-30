"""
SwapFace Detector using EfficientNet-B0
Binary classification: Real (0) vs Fake/SwapFace (1)
Compatible with DeepfakeBench framework
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from training.efficientnet_b0_backbone import EfficientNetB0Backbone

logger = logging.getLogger(__name__)


class SwapFaceDetector(nn.Module):
    """
    Lightweight SwapFace Detector using EfficientNet-B0
    Designed for mobile deployment (ONNX/TFLite conversion)
    """
    
    def __init__(self, config: Dict):
        super(SwapFaceDetector, self).__init__()
        self.config = config
        
        # Model parameters
        self.num_classes = config.get('num_classes', 2)
        self.dropout = config.get('dropout', 0.2)
        self.pretrained = config.get('pretrained', True)
        self.input_size = config.get('input_size', [224, 224])
        
        # Build backbone
        self.backbone = self.build_backbone(config)
        
        # Loss function
        self.loss_func = self.build_loss(config)
        
        # For metrics tracking
        self.prob = []
        self.label = []
        self.video_names = []
        self.correct = 0
        self.total = 0
        
        logger.info(f'SwapFaceDetector initialized with EfficientNet-B0')
        logger.info(f'Num classes: {self.num_classes}, Input size: {self.input_size}')
    
    def build_backbone(self, config: Dict) -> EfficientNetB0Backbone:
        """Build the EfficientNet-B0 backbone"""
        backbone = EfficientNetB0Backbone(
            num_classes=self.num_classes,
            dropout=self.dropout,
            pretrained=self.pretrained,
            input_channels=3
        )
        
        # Load custom pretrained weights if specified
        pretrained_path = config.get('pretrained_path', None)
        if pretrained_path and os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # Handle different state dict formats
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Load with strict=False to handle classifier mismatch
            missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logger.warning(f'Missing keys when loading pretrained: {missing_keys}')
            if unexpected_keys:
                logger.warning(f'Unexpected keys when loading pretrained: {unexpected_keys}')
            logger.info(f'Loaded pretrained weights from {pretrained_path}')
        elif self.pretrained:
            logger.info('Using ImageNet pretrained EfficientNet-B0 weights')
        else:
            logger.info('Training from scratch (no pretrained weights)')
        
        return backbone
    
    def build_loss(self, config: Dict) -> nn.Module:
        """Build loss function"""
        loss_type = config.get('loss_func', 'cross_entropy')
        label_smoothing = config.get('label_smoothing', 0.1)
        
        if loss_type == 'cross_entropy':
            return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        elif loss_type == 'bce':
            return nn.BCEWithLogitsLoss()
        elif loss_type == 'focal':
            return FocalLoss(alpha=config.get('focal_alpha', 0.25), 
                           gamma=config.get('focal_gamma', 2.0))
        else:
            raise ValueError(f'Unknown loss function: {loss_type}')
    
    def features(self, data_dict: Dict) -> torch.Tensor:
        """Extract features from backbone"""
        return self.backbone.features(data_dict['image'])
    
    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        """Apply classifier head"""
        return self.backbone.classifier_forward(features)
    
    def get_losses(self, data_dict: Dict, pred_dict: Dict) -> Dict:
        """Compute losses"""
        label = data_dict['label']
        pred = pred_dict['cls']
        
        # Handle label format
        if label.dim() > 1:
            label = label.squeeze()
        
        loss = self.loss_func(pred, label)
        return {'overall': loss}
    
    def get_train_metrics(self, data_dict: Dict, pred_dict: Dict) -> Dict:
        """Compute training metrics"""
        label = data_dict['label']
        pred = pred_dict['cls']
        
        if label.dim() > 1:
            label = label.squeeze()
        
        # Compute metrics
        with torch.no_grad():
            prob = F.softmax(pred, dim=1)[:, 1]
            pred_label = pred.argmax(dim=1)
            
            correct = (pred_label == label).sum().item()
            total = label.size(0)
            acc = correct / total if total > 0 else 0
            
            # AUC computation (simplified for batch)
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(label.cpu().numpy(), prob.cpu().numpy())
            except:
                auc = 0.0
            
            self.correct += correct
            self.total += total
        
        return {'acc': acc, 'auc': auc}
    
    def forward(self, data_dict: Dict, inference: bool = False) -> Dict:
        """Forward pass"""
        # Extract features
        features = self.features(data_dict)
        
        # Classification
        pred = self.classifier(features)
        
        # Get probabilities
        prob = F.softmax(pred, dim=1)[:, 1]  # Probability of fake class
        
        # Build prediction dict
        pred_dict = {
            'cls': pred,
            'prob': prob,
            'feat': features
        }
        
        return pred_dict
    
    def get_fake_score(self, data_dict: Dict) -> torch.Tensor:
        """Get fake score for inference (0-1)"""
        with torch.no_grad():
            pred_dict = self.forward(data_dict, inference=True)
            return pred_dict['prob']
    
    def reset_metrics(self):
        """Reset accumulated metrics"""
        self.prob = []
        self.label = []
        self.video_names = []
        self.correct = 0
        self.total = 0


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def create_swapface_detector(config: Dict) -> SwapFaceDetector:
    """Factory function to create SwapFaceDetector"""
    return SwapFaceDetector(config)


if __name__ == "__main__":
    # Test detector creation
    import logging
    logging.basicConfig(level=logging.INFO)
    
    config = {
        'num_classes': 2,
        'dropout': 0.2,
        'pretrained': True,
        'input_size': [224, 224],
        'loss_func': 'cross_entropy',
        'label_smoothing': 0.1
    }
    
    detector = create_swapface_detector(config)
    print(f"Detector created: {detector}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    data_dict = {'image': x, 'label': torch.tensor([0, 1])}
    
    with torch.no_grad():
        pred_dict = detector(data_dict)
    
    print(f"Output shape: {pred_dict['cls'].shape}")
    print(f"Prob shape: {pred_dict['prob'].shape}")
    print(f"Features shape: {pred_dict['feat'].shape}")
    print(f"Model parameters: {sum(p.numel() for p in detector.parameters()):,}")