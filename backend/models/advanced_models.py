"""
Advanced model architectures for medical imaging AI.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
import timm
from transformers import AutoModel, AutoConfig

logger = logging.getLogger(__name__)


class AttentionGate(nn.Module):
    """Attention gate for skip connections."""
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ResidualBlock(nn.Module):
    """Residual block with batch normalization."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
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


class AttentionUNet(nn.Module):
    """Attention U-Net for medical image segmentation."""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, 
                 base_features: int = 64):
        super(AttentionUNet, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, base_features)
        self.enc2 = self._conv_block(base_features, base_features * 2)
        self.enc3 = self._conv_block(base_features * 2, base_features * 4)
        self.enc4 = self._conv_block(base_features * 4, base_features * 8)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_features * 8, base_features * 16)
        
        # Attention gates
        self.att4 = AttentionGate(F_g=base_features * 16, F_l=base_features * 8, F_int=base_features * 4)
        self.att3 = AttentionGate(F_g=base_features * 8, F_l=base_features * 4, F_int=base_features * 2)
        self.att2 = AttentionGate(F_g=base_features * 4, F_l=base_features * 2, F_int=base_features)
        self.att1 = AttentionGate(F_g=base_features * 2, F_l=base_features, F_int=base_features // 2)
        
        # Decoder
        self.dec4 = self._conv_block(base_features * 16, base_features * 8)
        self.dec3 = self._conv_block(base_features * 8, base_features * 4)
        self.dec2 = self._conv_block(base_features * 4, base_features * 2)
        self.dec1 = self._conv_block(base_features * 2, base_features)
        
        # Final layer
        self.final = nn.Conv2d(base_features, out_channels, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(base_features * 16, base_features * 8, 2, 2)
        self.up3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, 2, 2)
        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, 2, 2)
        self.up1 = nn.ConvTranspose2d(base_features * 2, base_features, 2, 2)
    
    def _conv_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with attention
        d4 = self.up4(b)
        e4_att = self.att4(g=d4, x=e4)
        d4 = torch.cat((e4_att, d4), dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        e3_att = self.att3(g=d3, x=e3)
        d3 = torch.cat((e3_att, d3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        e2_att = self.att2(g=d2, x=e2)
        d2 = torch.cat((e2_att, d2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        e1_att = self.att1(g=d1, x=e1)
        d1 = torch.cat((e1_att, d1), dim=1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.final(d1))


class VisionTransformerUNet(nn.Module):
    """Vision Transformer U-Net for medical image segmentation."""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 1, out_channels: int = 1,
                 embed_dim: int = 768, num_heads: int = 12, 
                 num_layers: int = 12):
        super(VisionTransformerUNet, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, 
                                   kernel_size=patch_size, stride=patch_size)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, 2, 2),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, 2, 2),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, 2, 2),
            nn.BatchNorm2d(embed_dim // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 8, out_channels, kernel_size=1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer encoder
        x = self.transformer(x)
        
        # Remove class token and reshape
        x = x[:, 1:]  # Remove class token
        x = x.transpose(1, 2).reshape(B, -1, 
                                    self.img_size // self.patch_size, 
                                    self.img_size // self.patch_size)
        
        # Decoder
        x = self.decoder(x)
        
        return torch.sigmoid(x)


class EfficientNetUNet(nn.Module):
    """EfficientNet U-Net for medical image segmentation."""
    
    def __init__(self, model_name: str = 'efficientnet-b0', 
                 in_channels: int = 1, out_channels: int = 1):
        super(EfficientNetUNet, self).__init__()
        
        # Load pre-trained EfficientNet
        self.backbone = timm.create_model(model_name, pretrained=True, 
                                        in_chans=in_channels, features_only=True)
        
        # Get feature dimensions
        feature_dims = self.backbone.feature_info.channels()
        
        # Decoder
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(feature_dims[i], feature_dims[i-1] if i > 0 else 64, 2, 2),
                nn.BatchNorm2d(feature_dims[i-1] if i > 0 else 64),
                nn.ReLU(inplace=True)
            ) for i in range(len(feature_dims)-1, -1, -1)
        ])
        
        # Skip connections
        self.skip_convs = nn.ModuleList([
            nn.Conv2d(feature_dims[i], 64, 1) for i in range(len(feature_dims))
        ])
        
        # Final layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1)
        )
    
    def forward(self, x):
        # Encoder
        features = self.backbone(x)
        
        # Decoder with skip connections
        x = features[-1]
        for i, (decoder_layer, skip_conv) in enumerate(zip(self.decoder, self.skip_convs)):
            if i < len(features) - 1:
                skip = skip_conv(features[-(i+2)])
                x = decoder_layer(x)
                x = x + skip
            else:
                x = decoder_layer(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        return torch.sigmoid(x)


class EnsembleModel(nn.Module):
    """Ensemble of multiple models for improved performance."""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def forward(self, x):
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = torch.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += weight * pred
        
        return ensemble_pred


class MedicalVisionTransformer(nn.Module):
    """Medical-specific Vision Transformer for classification and detection."""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 1, num_classes: int = 2,
                 embed_dim: int = 768, num_heads: int = 12,
                 num_layers: int = 12, dropout: float = 0.1):
        super(MedicalVisionTransformer, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, 
                                   kernel_size=patch_size, stride=patch_size)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer encoder
        x = self.transformer(x)
        
        # Classification
        x = self.norm(x)
        cls_token = x[:, 0]
        logits = self.head(cls_token)
        
        return logits


class MultiTaskModel(nn.Module):
    """Multi-task model for simultaneous classification and segmentation."""
    
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super(MultiTaskModel, self).__init__()
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Segmentation head
        self.segmentation_head = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 1, 1)
        )
    
    def forward(self, x):
        # Shared features
        features = self.encoder(x)
        
        # Classification
        cls_logits = self.classifier(features)
        
        # Segmentation
        seg_logits = self.segmentation_head(features)
        
        return {
            'classification': cls_logits,
            'segmentation': torch.sigmoid(seg_logits)
        }


def create_model(model_type: str, **kwargs) -> nn.Module:
    """Factory function to create models."""
    
    models = {
        'attention_unet': AttentionUNet,
        'vit_unet': VisionTransformerUNet,
        'efficientnet_unet': EfficientNetUNet,
        'medical_vit': MedicalVisionTransformer,
        'multi_task': MultiTaskModel
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type](**kwargs)


def create_ensemble(models_config: List[Dict]) -> EnsembleModel:
    """Create an ensemble model from multiple configurations."""
    
    models = []
    for config in models_config:
        model = create_model(config['type'], **config['params'])
        models.append(model)
    
    weights = [config.get('weight', 1.0) for config in models_config]
    return EnsembleModel(models, weights)
