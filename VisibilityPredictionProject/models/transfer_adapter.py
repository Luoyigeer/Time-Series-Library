"""
Transfer Learning Adapter for Visibility Prediction
===================================================

This module implements transfer learning adapters to enable knowledge transfer
from multi-station GNN models to single-image visibility detection models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAlignmentLayer(nn.Module):
    """
    Layer to align features from different domains.
    
    Args:
        source_dim: Dimension of source features (from GNN)
        target_dim: Dimension of target features (from vision model)
        hidden_dim: Dimension of hidden layer
    """
    
    def __init__(self, source_dim, target_dim, hidden_dim=256):
        super(FeatureAlignmentLayer, self).__init__()
        
        self.alignment = nn.Sequential(
            nn.Linear(source_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, target_dim),
            nn.LayerNorm(target_dim)
        )
    
    def forward(self, source_features):
        """Align source features to target domain."""
        return self.alignment(source_features)


class DomainDiscriminator(nn.Module):
    """
    Domain discriminator for adversarial domain adaptation.
    
    Args:
        feature_dim: Dimension of input features
    """
    
    def __init__(self, feature_dim):
        super(DomainDiscriminator, self).__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # Binary classification: source vs target
        )
    
    def forward(self, features):
        """Classify domain of features."""
        return self.discriminator(features)


class TransferLearningAdapter(nn.Module):
    """
    Transfer learning adapter for bridging GNN and vision models.
    
    This adapter enables transfer of learned representations from multi-station
    visibility prediction (using GNN) to single-image visibility detection
    (using vision models). It uses feature alignment and optional adversarial
    training for domain adaptation.
    
    Args:
        gnn_model: Pre-trained GNN model
        vision_model: Vision model for image-based prediction
        gnn_feature_dim: Feature dimension from GNN model
        vision_feature_dim: Feature dimension from vision model
        use_adversarial: Whether to use adversarial domain adaptation
        freeze_gnn: Whether to freeze GNN weights during transfer
        dropout: Dropout rate
    """
    
    def __init__(self, gnn_model, vision_model, 
                 gnn_feature_dim=128, vision_feature_dim=128,
                 use_adversarial=True, freeze_gnn=True, dropout=0.2):
        super(TransferLearningAdapter, self).__init__()
        
        self.gnn_model = gnn_model
        self.vision_model = vision_model
        self.use_adversarial = use_adversarial
        
        # Freeze GNN if requested
        if freeze_gnn:
            for param in self.gnn_model.parameters():
                param.requires_grad = False
        
        # Feature alignment layers
        self.gnn_to_common = FeatureAlignmentLayer(
            gnn_feature_dim, 256, hidden_dim=256
        )
        self.vision_to_common = FeatureAlignmentLayer(
            vision_feature_dim, 256, hidden_dim=256
        )
        
        # Domain discriminator for adversarial training
        if use_adversarial:
            self.domain_discriminator = DomainDiscriminator(256)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),  # Concatenated features
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final prediction layer
        self.predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Single visibility value
        )
    
    def forward(self, gnn_input=None, image_input=None, 
                mode='joint', return_domain_logits=False):
        """
        Forward pass with multiple modes.
        
        Args:
            gnn_input: Input for GNN model [batch_size, seq_len, num_stations, input_dim]
            image_input: Input images [batch_size, 3, height, width]
            mode: Forward mode - 'joint', 'gnn_only', 'vision_only'
            return_domain_logits: Whether to return domain discrimination logits
            
        Returns:
            predictions: Visibility predictions [batch_size, 1]
            (optional) domain_logits: Domain classification logits
        """
        domain_logits = None
        
        if mode == 'joint':
            # Process both GNN and vision inputs
            assert gnn_input is not None and image_input is not None
            
            # Extract GNN features
            with torch.no_grad() if not self.training else torch.enable_grad():
                gnn_output = self.gnn_model(gnn_input)  # [batch_size, num_stations, 1]
                gnn_features = gnn_output.mean(dim=1)  # Average over stations
            
            # Extract vision features
            _, vision_features = self.vision_model(image_input, return_features=True)
            
            # Align features to common space
            gnn_aligned = self.gnn_to_common(gnn_features)
            vision_aligned = self.vision_to_common(vision_features)
            
            # Domain discrimination if requested
            if self.use_adversarial and return_domain_logits:
                gnn_domain = self.domain_discriminator(gnn_aligned)
                vision_domain = self.domain_discriminator(vision_aligned)
                domain_logits = (gnn_domain, vision_domain)
            
            # Fuse features
            combined = torch.cat([gnn_aligned, vision_aligned], dim=-1)
            fused = self.fusion(combined)
            
            # Generate prediction
            predictions = self.predictor(fused)
            
        elif mode == 'gnn_only':
            # Use only GNN input
            assert gnn_input is not None
            
            with torch.no_grad() if not self.training else torch.enable_grad():
                gnn_output = self.gnn_model(gnn_input)
                gnn_features = gnn_output.mean(dim=1)
            
            gnn_aligned = self.gnn_to_common(gnn_features)
            
            # Use zero features for vision pathway
            vision_aligned = torch.zeros_like(gnn_aligned)
            
            combined = torch.cat([gnn_aligned, vision_aligned], dim=-1)
            fused = self.fusion(combined)
            predictions = self.predictor(fused)
            
        elif mode == 'vision_only':
            # Use only vision input
            assert image_input is not None
            
            _, vision_features = self.vision_model(image_input, return_features=True)
            vision_aligned = self.vision_to_common(vision_features)
            
            # Use zero features for GNN pathway
            gnn_aligned = torch.zeros_like(vision_aligned)
            
            combined = torch.cat([gnn_aligned, vision_aligned], dim=-1)
            fused = self.fusion(combined)
            predictions = self.predictor(fused)
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        if return_domain_logits:
            return predictions, domain_logits
        return predictions
    
    def fine_tune_vision(self, unfreeze_layers=None):
        """
        Fine-tune vision model layers.
        
        Args:
            unfreeze_layers: List of layer names to unfreeze, or None for all
        """
        if unfreeze_layers is None:
            # Unfreeze all vision model parameters
            for param in self.vision_model.parameters():
                param.requires_grad = True
        else:
            # Unfreeze specific layers
            for name, param in self.vision_model.named_parameters():
                if any(layer in name for layer in unfreeze_layers):
                    param.requires_grad = True
    
    def get_aligned_features(self, gnn_input=None, image_input=None):
        """
        Extract aligned features without prediction.
        
        Args:
            gnn_input: Input for GNN model
            image_input: Input images
            
        Returns:
            Aligned features in common space
        """
        features = []
        
        if gnn_input is not None:
            with torch.no_grad():
                gnn_output = self.gnn_model(gnn_input)
                gnn_features = gnn_output.mean(dim=1)
                gnn_aligned = self.gnn_to_common(gnn_features)
                features.append(gnn_aligned)
        
        if image_input is not None:
            with torch.no_grad():
                _, vision_features = self.vision_model(image_input, return_features=True)
                vision_aligned = self.vision_to_common(vision_features)
                features.append(vision_aligned)
        
        return features
