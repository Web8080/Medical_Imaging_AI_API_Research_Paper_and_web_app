"""
Research Paper Model Implementations
Based on the specific methodologies described in the research paper.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

# Medical imaging libraries
try:
    import monai
    from monai.networks.nets import UNet, DynUNet, AttentionUnet
    from monai.networks.blocks import Convolution, ResidualUnit, AttentionBlock
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    logging.warning("MONAI not available. Some models will use fallback implementations.")

try:
    import torchvision.models as models
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    logging.warning("torchvision not available. Mask R-CNN will use fallback implementation.")

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logging.warning("timm not available. Vision Transformer will use fallback implementation.")

logger = logging.getLogger(__name__)


class ResearchPaperModels:
    """
    Model implementations following the research paper methodology.
    
    Implements:
    1. U-Net Variants (Standard U-Net, 3D U-Net, Attention U-Net)
    2. nnU-Net (Self-configuring framework)
    3. Mask R-CNN (For detection tasks)
    4. Vision Transformers (Transformer-based architectures)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
    
    def create_unet_2d(self, model_config: Dict[str, Any]) -> nn.Module:
        """Create 2D U-Net as specified in research paper."""
        if MONAI_AVAILABLE:
            return UNet(
                spatial_dims=2,
                in_channels=model_config.get('in_channels', 1),
                out_channels=model_config.get('out_channels', 1),
                channels=model_config.get('channels', [16, 32, 64, 128, 256]),
                strides=model_config.get('strides', [2, 2, 2, 2]),
                num_res_units=model_config.get('num_res_units', 2),
            )
        else:
            return self._create_fallback_unet_2d(model_config)
    
    def create_unet_3d(self, model_config: Dict[str, Any]) -> nn.Module:
        """Create 3D U-Net as specified in research paper."""
        if MONAI_AVAILABLE:
            return UNet(
                spatial_dims=3,
                in_channels=model_config.get('in_channels', 1),
                out_channels=model_config.get('out_channels', 1),
                channels=model_config.get('channels', [16, 32, 64, 128, 256]),
                strides=model_config.get('strides', [2, 2, 2, 2]),
                num_res_units=model_config.get('num_res_units', 2),
            )
        else:
            return self._create_fallback_unet_3d(model_config)
    
    def create_attention_unet(self, model_config: Dict[str, Any]) -> nn.Module:
        """Create Attention U-Net as specified in research paper."""
        if MONAI_AVAILABLE:
            return AttentionUnet(
                spatial_dims=3,
                in_channels=model_config.get('in_channels', 1),
                out_channels=model_config.get('out_channels', 1),
                channels=model_config.get('channels', [16, 32, 64, 128, 256]),
                strides=model_config.get('strides', [2, 2, 2, 2]),
                num_res_units=model_config.get('num_res_units', 2),
            )
        else:
            return self._create_fallback_attention_unet(model_config)
    
    def create_nnunet(self, model_config: Dict[str, Any]) -> nn.Module:
        """Create nnU-Net as specified in research paper."""
        if MONAI_AVAILABLE:
            return DynUNet(
                spatial_dims=3,
                in_channels=model_config.get('in_channels', 1),
                out_channels=model_config.get('out_channels', 1),
                kernel_size=model_config.get('kernel_size', [3, 3, 3, 3, 3]),
                strides=model_config.get('strides', [1, 2, 2, 2, 2]),
                upsample_kernel_size=model_config.get('upsample_kernel_size', [2, 2, 2, 2]),
                filters=model_config.get('filters', [32, 64, 128, 256, 512]),
                norm_name=model_config.get('norm_name', 'instance'),
                act_name=model_config.get('act_name', 'leakyrelu'),
                deep_supervision=model_config.get('deep_supervision', True),
                deep_supr_num=model_config.get('deep_supr_num', 3),
            )
        else:
            return self._create_fallback_nnunet(model_config)
    
    def create_mask_rcnn(self, model_config: Dict[str, Any]) -> nn.Module:
        """Create Mask R-CNN as specified in research paper."""
        if TORCHVISION_AVAILABLE:
            # Load pre-trained Mask R-CNN
            model = maskrcnn_resnet50_fpn(pretrained=True)
            
            # Modify for medical imaging
            model.backbone.body.conv1 = nn.Conv2d(
                model_config.get('in_channels', 1), 64, 
                kernel_size=7, stride=2, padding=3, bias=False
            )
            
            # Update number of classes
            num_classes = model_config.get('num_classes', 2)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = nn.Sequential(
                nn.Linear(in_features, 1024),
                nn.ReLU(),
                nn.Linear(1024, num_classes * 4),  # 4 coordinates per class
            )
            
            # Update mask predictor
            in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
            model.roi_heads.mask_predictor.conv5_mask = nn.Conv2d(
                in_features_mask, 256, kernel_size=3, stride=1, padding=1
            )
            model.roi_heads.mask_predictor.mask_fcn_logits = nn.Conv2d(
                256, num_classes, kernel_size=1, stride=1
            )
            
            return model
        else:
            return self._create_fallback_mask_rcnn(model_config)
    
    def create_vision_transformer(self, model_config: Dict[str, Any]) -> nn.Module:
        """Create Vision Transformer as specified in research paper."""
        if TIMM_AVAILABLE:
            # Use timm's Vision Transformer
            model = timm.create_model(
                'vit_base_patch16_224',
                pretrained=True,
                in_chans=model_config.get('in_channels', 1),
                num_classes=model_config.get('num_classes', 2)
            )
            return model
        else:
            return self._create_fallback_vision_transformer(model_config)
    
    def _create_fallback_unet_2d(self, model_config: Dict[str, Any]) -> nn.Module:
        """Fallback 2D U-Net implementation."""
        return UNet2D(
            in_channels=model_config.get('in_channels', 1),
            out_channels=model_config.get('out_channels', 1),
            base_features=model_config.get('base_features', 64)
        )
    
    def _create_fallback_unet_3d(self, model_config: Dict[str, Any]) -> nn.Module:
        """Fallback 3D U-Net implementation."""
        return UNet3D(
            in_channels=model_config.get('in_channels', 1),
            out_channels=model_config.get('out_channels', 1),
            base_features=model_config.get('base_features', 64)
        )
    
    def _create_fallback_attention_unet(self, model_config: Dict[str, Any]) -> nn.Module:
        """Fallback Attention U-Net implementation."""
        return AttentionUNet3D(
            in_channels=model_config.get('in_channels', 1),
            out_channels=model_config.get('out_channels', 1),
            base_features=model_config.get('base_features', 64)
        )
    
    def _create_fallback_nnunet(self, model_config: Dict[str, Any]) -> nn.Module:
        """Fallback nnU-Net implementation."""
        return nnUNet3D(
            in_channels=model_config.get('in_channels', 1),
            out_channels=model_config.get('out_channels', 1),
            base_features=model_config.get('base_features', 32)
        )
    
    def _create_fallback_mask_rcnn(self, model_config: Dict[str, Any]) -> nn.Module:
        """Fallback Mask R-CNN implementation."""
        return MedicalMaskRCNN(
            in_channels=model_config.get('in_channels', 1),
            num_classes=model_config.get('num_classes', 2)
        )
    
    def _create_fallback_vision_transformer(self, model_config: Dict[str, Any]) -> nn.Module:
        """Fallback Vision Transformer implementation."""
        return MedicalVisionTransformer(
            img_size=model_config.get('img_size', 224),
            patch_size=model_config.get('patch_size', 16),
            in_channels=model_config.get('in_channels', 1),
            num_classes=model_config.get('num_classes', 2),
            embed_dim=model_config.get('embed_dim', 768),
            num_heads=model_config.get('num_heads', 12),
            num_layers=model_config.get('num_layers', 12)
        )


# Fallback Model Implementations

class UNet2D(nn.Module):
    """Fallback 2D U-Net implementation."""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_features: int = 64):
        super(UNet2D, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, base_features)
        self.enc2 = self._conv_block(base_features, base_features * 2)
        self.enc3 = self._conv_block(base_features * 2, base_features * 4)
        self.enc4 = self._conv_block(base_features * 4, base_features * 8)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_features * 8, base_features * 16)
        
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
        
        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.final(d1))


class UNet3D(nn.Module):
    """Fallback 3D U-Net implementation."""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_features: int = 64):
        super(UNet3D, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, base_features)
        self.enc2 = self._conv_block(base_features, base_features * 2)
        self.enc3 = self._conv_block(base_features * 2, base_features * 4)
        self.enc4 = self._conv_block(base_features * 4, base_features * 8)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_features * 8, base_features * 16)
        
        # Decoder
        self.dec4 = self._conv_block(base_features * 16, base_features * 8)
        self.dec3 = self._conv_block(base_features * 8, base_features * 4)
        self.dec2 = self._conv_block(base_features * 4, base_features * 2)
        self.dec1 = self._conv_block(base_features * 2, base_features)
        
        # Final layer
        self.final = nn.Conv3d(base_features, out_channels, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose3d(base_features * 16, base_features * 8, 2, 2)
        self.up3 = nn.ConvTranspose3d(base_features * 8, base_features * 4, 2, 2)
        self.up2 = nn.ConvTranspose3d(base_features * 4, base_features * 2, 2, 2)
        self.up1 = nn.ConvTranspose3d(base_features * 2, base_features, 2, 2)
    
    def _conv_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
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
        
        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.final(d1))


class AttentionUNet3D(nn.Module):
    """Fallback 3D Attention U-Net implementation."""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_features: int = 64):
        super(AttentionUNet3D, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, base_features)
        self.enc2 = self._conv_block(base_features, base_features * 2)
        self.enc3 = self._conv_block(base_features * 2, base_features * 4)
        self.enc4 = self._conv_block(base_features * 4, base_features * 8)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_features * 8, base_features * 16)
        
        # Attention gates
        self.att4 = AttentionGate3D(base_features * 16, base_features * 8, base_features * 4)
        self.att3 = AttentionGate3D(base_features * 8, base_features * 4, base_features * 2)
        self.att2 = AttentionGate3D(base_features * 4, base_features * 2, base_features)
        self.att1 = AttentionGate3D(base_features * 2, base_features, base_features // 2)
        
        # Decoder
        self.dec4 = self._conv_block(base_features * 16, base_features * 8)
        self.dec3 = self._conv_block(base_features * 8, base_features * 4)
        self.dec2 = self._conv_block(base_features * 4, base_features * 2)
        self.dec1 = self._conv_block(base_features * 2, base_features)
        
        # Final layer
        self.final = nn.Conv3d(base_features, out_channels, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose3d(base_features * 16, base_features * 8, 2, 2)
        self.up3 = nn.ConvTranspose3d(base_features * 8, base_features * 4, 2, 2)
        self.up2 = nn.ConvTranspose3d(base_features * 4, base_features * 2, 2, 2)
        self.up1 = nn.ConvTranspose3d(base_features * 2, base_features, 2, 2)
    
    def _conv_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
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


class AttentionGate3D(nn.Module):
    """3D Attention gate for skip connections."""
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionGate3D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class nnUNet3D(nn.Module):
    """Fallback nnU-Net implementation."""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_features: int = 32):
        super(nnUNet3D, self).__init__()
        
        # Simplified nnU-Net architecture
        self.encoder = nn.ModuleList([
            self._conv_block(in_channels, base_features),
            self._conv_block(base_features, base_features * 2),
            self._conv_block(base_features * 2, base_features * 4),
            self._conv_block(base_features * 4, base_features * 8),
        ])
        
        self.bottleneck = self._conv_block(base_features * 8, base_features * 16)
        
        self.decoder = nn.ModuleList([
            self._conv_block(base_features * 16, base_features * 8),
            self._conv_block(base_features * 8, base_features * 4),
            self._conv_block(base_features * 4, base_features * 2),
            self._conv_block(base_features * 2, base_features),
        ])
        
        self.final = nn.Conv3d(base_features, out_channels, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.upsample = nn.ModuleList([
            nn.ConvTranspose3d(base_features * 16, base_features * 8, 2, 2),
            nn.ConvTranspose3d(base_features * 8, base_features * 4, 2, 2),
            nn.ConvTranspose3d(base_features * 4, base_features * 2, 2, 2),
            nn.ConvTranspose3d(base_features * 2, base_features, 2, 2),
        ])
    
    def _conv_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        encoder_outputs = []
        for encoder in self.encoder:
            x = encoder(x)
            encoder_outputs.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for i, (decoder, upsample) in enumerate(zip(self.decoder, self.upsample)):
            x = upsample(x)
            x = torch.cat((x, encoder_outputs[-(i+1)]), dim=1)
            x = decoder(x)
        
        return torch.sigmoid(self.final(x))


class MedicalMaskRCNN(nn.Module):
    """Fallback Mask R-CNN implementation for medical imaging."""
    
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super(MedicalMaskRCNN, self).__init__()
        
        # Simplified backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Detection head
        self.detection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes * 4),  # 4 coordinates per class
        )
        
        # Mask head
        self.mask_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1, 1, 0),
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        # Detection
        detection = self.detection_head(features)
        
        # Mask
        mask = self.mask_head(features)
        
        return {
            'detection': detection,
            'mask': torch.sigmoid(mask)
        }


class MedicalVisionTransformer(nn.Module):
    """Fallback Vision Transformer implementation for medical imaging."""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 1, num_classes: int = 2,
                 embed_dim: int = 768, num_heads: int = 12, num_layers: int = 12):
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
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True
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
        
        # Transformer encoder
        x = self.transformer(x)
        
        # Classification
        x = self.norm(x)
        cls_token = x[:, 0]
        logits = self.head(cls_token)
        
        return logits
