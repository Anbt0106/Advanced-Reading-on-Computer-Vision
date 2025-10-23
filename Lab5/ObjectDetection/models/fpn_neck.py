# models/fpn_neck.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from Lab5.CNN_ELAN_Class import ConvBNAct


class SimpleFPN(nn.Module):
    """Đơn giản FPN cho 3 scales"""

    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()

        # 1x1 conv để match channels
        self.lateral_convs = nn.ModuleList([
            ConvBNAct(ch, out_channels, k=1, s=1, p=0)
            for ch in in_channels_list
        ])

        # 3x3 conv sau upsampling
        self.fpn_convs = nn.ModuleList([
            ConvBNAct(out_channels, out_channels, k=3, s=1, p=1)
            for _ in in_channels_list
        ])

    def forward(self, features):
        """
        Args:
            features: [P1/4, P2/8, P3/16]
        Returns:
            enhanced_features: [FPN1/4, FPN2/8, FPN3/16]
        """
        # Lateral connections
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample higher level và add
            upsampled = F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode='nearest'
            )
            laterals[i - 1] = laterals[i - 1] + upsampled

        # Final convs
        outputs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]

        return outputs