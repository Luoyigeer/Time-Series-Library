"""
Vision Network for Single-Image Visibility Detection
====================================================

This module implements a CNN-based model for estimating visibility from
single images, which can be used with transfer learning from the GNN models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block for feature extraction.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for convolution
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class AttentionModule(nn.Module):
    """
    Attention module to focus on relevant image regions.
    
    Args:
        in_channels: Number of input channels
    """
    
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        channel_attn = self.channel_attention(x)
        x = x * channel_attn
        
        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_attn = self.spatial_attention(spatial_input)
        x = x * spatial_attn
        
        return x


class VisibilityVisionNet(nn.Module):
    """
    CNN-based model for visibility detection from images.
    
    This model extracts visual features from images to estimate visibility
    conditions. It can be used standalone or as part of a transfer learning
    pipeline from the multi-station GNN models.
    
    Args:
        num_classes: Number of visibility classes (default: 1 for regression)
        pretrained: Whether to use pretrained backbone
        dropout: Dropout rate
    """
    
    def __init__(self, num_classes=1, pretrained=False, dropout=0.3):
        super(VisibilityVisionNet, self).__init__()
        
        self.num_classes = num_classes
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)
        
        # Attention modules
        self.attention1 = AttentionModule(64)
        self.attention2 = AttentionModule(128)
        self.attention3 = AttentionModule(256)
        self.attention4 = AttentionModule(512)
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature extraction head
        self.feature_extractor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128)
        )
        
        # Classification/Regression head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        if pretrained:
            self._load_pretrained_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Create a layer with multiple residual blocks."""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def _load_pretrained_weights(self):
        """Load pretrained weights if available."""
        # Placeholder for pretrained weight loading
        # In practice, this would load weights from a pretrained model
        pass
    
    def forward(self, x, return_features=False):
        """
        Forward pass for visibility prediction from images.
        
        Args:
            x: Input images [batch_size, 3, height, width]
            return_features: If True, return intermediate features
            
        Returns:
            If return_features=False: predictions [batch_size, num_classes]
            If return_features=True: (predictions, features) tuple
        """
        # Initial convolution
        x = self.conv1(x)
        
        # Residual blocks with attention
        x = self.layer1(x)
        x = self.attention1(x)
        
        x = self.layer2(x)
        x = self.attention2(x)
        
        x = self.layer3(x)
        x = self.attention3(x)
        
        x = self.layer4(x)
        x = self.attention4(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Generate predictions
        predictions = self.classifier(features)
        
        if return_features:
            return predictions, features
        return predictions
    
    def extract_features(self, x):
        """
        Extract features without classification.
        
        Args:
            x: Input images [batch_size, 3, height, width]
            
        Returns:
            Features [batch_size, 128]
        """
        with torch.no_grad():
            _, features = self.forward(x, return_features=True)
        return features
