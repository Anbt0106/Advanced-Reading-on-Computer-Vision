# models/eelan_detector.py
import torch
import torch.nn as nn
from .eelan_backbone import EELANBackbone
from .fpn_neck import SimpleFPN
from .yolo_head import YOLOHead


class EELANDetector(nn.Module):
    """Complete Object Detection Model"""

    def __init__(self, num_classes=20, C_stem=32, C_stage=96):
        super().__init__()

        self.num_classes = num_classes

        # Components
        self.backbone = EELANBackbone(C_stem, C_stage)
        self.neck = SimpleFPN(self.backbone.out_channels, out_channels=256)
        self.heads = nn.ModuleList([
            YOLOHead(num_classes, 256, num_anchors=3)
            for _ in range(3)
        ])

        # Anchors (width, height) for each scale
        self.anchors = torch.tensor([
            [[10, 13], [16, 30], [33, 23]],  # Scale 1 (/4)
            [[30, 61], [62, 45], [59, 119]],  # Scale 2 (/8)
            [[116, 90], [156, 198], [373, 326]]  # Scale 3 (/16)
        ], dtype=torch.float32)

        self.strides = torch.tensor([4, 8, 16], dtype=torch.float32)

    def forward(self, x):
        # Feature extraction
        backbone_features = self.backbone(x)  # 3 scales
        fpn_features = self.neck(backbone_features)

        # Detection heads
        predictions = []
        for feat, head in zip(fpn_features, self.heads):
            pred = head(feat)
            predictions.append(pred)

        return predictions

    def load_classification_backbone(self, ckpt_path):
        """Load pretrained classification weights"""
        self.backbone.load_classification_weights(ckpt_path)
        print("Loaded classification backbone weights")