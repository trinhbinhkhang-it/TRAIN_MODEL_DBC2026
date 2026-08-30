"""
EfficientNet-B0 Backbone for SwapFace Detection
Lightweight backbone optimized for mobile deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class EfficientNetB0Backbone(nn.Module):
    """EfficientNet-B0 backbone with customizable classifier head"""
    
    def __init__(self, num_classes=2, dropout=0.2, pretrained=True, input_channels=3):
        super(EfficientNetB0Backbone, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.input_channels = input_channels
        
        # Load EfficientNet-B0
        if pretrained:
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            self.backbone = EfficientNet.from_name('efficientnet-b0')
        
        # Modify stem for different input channels
        if input_channels != 3:
            self.backbone._conv_stem = nn.Conv2d(
                input_channels, 32, kernel_size=3, stride=2, bias=False
            )
        
        # Remove original classifier
        self.backbone._fc = nn.Identity()
        self.backbone._dropout = nn.Identity()
        
        # Feature dimension for EfficientNet-B0
        self.feature_dim = 1280
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier()
    
    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def features(self, x):
        """Extract features from backbone"""
        x = self.backbone.extract_features(x)
        return x
    
    def classifier_forward(self, features):
        """Apply classifier head to features"""
        x = F.adaptive_avg_pool2d(features, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def forward(self, x):
        """Full forward pass"""
        features = self.features(x)
        output = self.classifier_forward(features)
        return output
    
    def get_feature_dim(self):
        return self.feature_dim


def create_efficientnet_b0(num_classes=2, dropout=0.2, pretrained=True):
    """Factory function to create EfficientNet-B0 model"""
    return EfficientNetB0Backbone(
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained
    )


if __name__ == "__main__":
    # Test model creation
    model = create_efficientnet_b0(num_classes=2, dropout=0.2, pretrained=True)
    print(f"Model created: {model}")
    print(f"Feature dim: {model.get_feature_dim()}")
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")