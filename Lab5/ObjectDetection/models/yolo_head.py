# models/yolo_head.py
import torch
import torch.nn as nn

from Lab5.CNN_ELAN_Class import ConvBNAct


class YOLOHead(nn.Module):
    """YOLO Detection Head"""

    def __init__(self, num_classes, in_channels=256, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Output: (4_bbox + 1_obj + num_classes) * num_anchors
        out_ch = num_anchors * (5 + num_classes)

        self.conv = nn.Sequential(
            ConvBNAct(in_channels, in_channels, k=3, s=1, p=1),
            nn.Conv2d(in_channels, out_ch, kernel_size=1)
        )

    def forward(self, x):
        """
        Returns: [B, num_anchors, H, W, 5+num_classes]
        """
        B, _, H, W = x.shape

        pred = self.conv(x)  # [B, num_anchors*(5+nc), H, W]

        # Reshape
        pred = pred.view(B, self.num_anchors, -1, H, W)
        pred = pred.permute(0, 1, 3, 4, 2)  # [B, na, H, W, 5+nc]

        return pred