# models/eelan_backbone.py
import torch
import torch.nn as nn
from typing import List

# Import từ classification model đã có
from Lab5.CNN_ELAN_Class import EELANClassifier, ConvBNAct, EELANLite


class EELANBackbone(nn.Module):
    """EELAN Backbone cho Object Detection"""

    def __init__(self, C_stem=32, C_stage=96, m=2.0, g=2):
        super().__init__()

        # Stages từ EELANClassifier
        self.stem = nn.Sequential(
            ConvBNAct(3, C_stem, k=3, s=2),  # /2
            ConvBNAct(C_stem, C_stem, k=3, s=1),
        )

        self.stage1 = nn.Sequential(
            nn.MaxPool2d(2, 2),  # /4
            ConvBNAct(C_stem, C_stage, k=3, s=1),
            EELANLite(C_stage, C_stage, m=m, g=g, use_skip=True)
        )

        self.stage2 = nn.Sequential(
            nn.MaxPool2d(2, 2),  # /8
            ConvBNAct(C_stage, C_stage * 2, k=3, s=1),
            EELANLite(C_stage * 2, C_stage * 2, m=m, g=g, use_skip=True)
        )

        self.stage3 = nn.Sequential(
            nn.MaxPool2d(2, 2),  # /16
            ConvBNAct(C_stage * 2, C_stage * 4, k=3, s=1),
            EELANLite(C_stage * 4, C_stage * 4, m=m, g=g, use_skip=True)
        )

        # Output channels
        self.out_channels = [C_stage, C_stage * 2, C_stage * 4]

    def forward(self, x):
        """Return multi-scale features"""
        x = self.stem(x)  # /2

        p1 = self.stage1(x)  # /4
        p2 = self.stage2(p1)  # /8
        p3 = self.stage3(p2)  # /16

        return [p1, p2, p3]  # 3 scales

    def load_classification_weights(self, ckpt_path):
        """Load weights từ classification model"""
        ckpt = torch.load(ckpt_path, map_location='cpu')
        clf_state = ckpt['model_state_dict']

        # Map classification weights to backbone
        backbone_state = {}
        for name, param in clf_state.items():
            if name.startswith(('stem', 'stage1', 'eelan')):
                # Map eelan -> stage1.eelan
                if name.startswith('eelan'):
                    new_name = 'stage1.' + name
                else:
                    new_name = name
                backbone_state[new_name] = param

        # Load matched weights
        self.load_state_dict(backbone_state, strict=False)
        print(f"Loaded classification weights: {len(backbone_state)} layers")