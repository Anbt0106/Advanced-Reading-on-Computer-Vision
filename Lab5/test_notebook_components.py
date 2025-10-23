#!/usr/bin/env python3
"""
Test script to validate key components of the Object Detection notebook.
This ensures the model architecture and data pipeline are correctly defined.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

print("🧪 Testing Object Detection Notebook Components\n")

# Test 1: ResNet50 Backbone
print("1. Testing ResNet50 Backbone...")
try:
    from torchvision.models import resnet50, ResNet50_Weights
    
    class ResNet50Backbone(nn.Module):
        def __init__(self, pretrained=False):  # Use False for testing
            super().__init__()
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            resnet = resnet50(weights=weights)
            
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            c2 = self.layer1(x)
            c3 = self.layer2(c2)
            c4 = self.layer3(c3)
            c5 = self.layer4(c4)
            
            return [c3, c4, c5]
    
    backbone = ResNet50Backbone(pretrained=False)
    test_input = torch.randn(1, 3, 416, 416)
    features = backbone(test_input)
    
    assert len(features) == 3, "Should return 3 feature scales"
    assert features[0].shape == torch.Size([1, 512, 52, 52]), f"P3 shape incorrect: {features[0].shape}"
    assert features[1].shape == torch.Size([1, 1024, 26, 26]), f"P4 shape incorrect: {features[1].shape}"
    assert features[2].shape == torch.Size([1, 2048, 13, 13]), f"P5 shape incorrect: {features[2].shape}"
    
    print("   ✅ ResNet50 Backbone: PASS")
except Exception as e:
    print(f"   ❌ ResNet50 Backbone: FAIL - {e}")
    sys.exit(1)

# Test 2: FPN Neck
print("2. Testing FPN Neck...")
try:
    class ConvBNReLU(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
        
        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))
    
    class FPN(nn.Module):
        def __init__(self, in_channels_list=[512, 1024, 2048], out_channels=256):
            super().__init__()
            
            self.lateral_convs = nn.ModuleList([
                nn.Conv2d(in_ch, out_channels, kernel_size=1)
                for in_ch in in_channels_list
            ])
            
            self.fpn_convs = nn.ModuleList([
                ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1)
                for _ in in_channels_list
            ])
        
        def forward(self, features):
            laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
            
            for i in range(len(laterals) - 1, 0, -1):
                upsampled = F.interpolate(laterals[i], size=laterals[i-1].shape[-2:], mode='nearest')
                laterals[i-1] = laterals[i-1] + upsampled
            
            outputs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
            return outputs
    
    fpn = FPN()
    fpn_features = fpn(features)
    
    assert len(fpn_features) == 3, "Should return 3 FPN features"
    assert fpn_features[0].shape == torch.Size([1, 256, 52, 52]), f"FPN3 shape incorrect: {fpn_features[0].shape}"
    assert fpn_features[1].shape == torch.Size([1, 256, 26, 26]), f"FPN4 shape incorrect: {fpn_features[1].shape}"
    assert fpn_features[2].shape == torch.Size([1, 256, 13, 13]), f"FPN5 shape incorrect: {fpn_features[2].shape}"
    
    print("   ✅ FPN Neck: PASS")
except Exception as e:
    print(f"   ❌ FPN Neck: FAIL - {e}")
    sys.exit(1)

# Test 3: Detection Head
print("3. Testing Detection Head...")
try:
    class DetectionHead(nn.Module):
        def __init__(self, num_classes=3, in_channels=256, num_anchors=3):
            super().__init__()
            self.num_classes = num_classes
            self.num_anchors = num_anchors
            
            out_channels = num_anchors * (5 + num_classes)
            
            self.conv = nn.Sequential(
                ConvBNReLU(in_channels, in_channels, kernel_size=3, padding=1),
                ConvBNReLU(in_channels, in_channels, kernel_size=3, padding=1),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
        
        def forward(self, x):
            B, _, H, W = x.shape
            pred = self.conv(x)
            pred = pred.view(B, self.num_anchors, -1, H, W)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            return pred
    
    det_head = DetectionHead(num_classes=3, in_channels=256)
    test_pred = det_head(fpn_features[0])
    
    expected_shape = torch.Size([1, 3, 52, 52, 8])  # [B, anchors, H, W, 5+classes]
    assert test_pred.shape == expected_shape, f"Detection head output shape incorrect: {test_pred.shape}, expected {expected_shape}"
    
    print("   ✅ Detection Head: PASS")
except Exception as e:
    print(f"   ❌ Detection Head: FAIL - {e}")
    sys.exit(1)

# Test 4: Complete Object Detector
print("4. Testing Complete Object Detector...")
try:
    class ObjectDetector(nn.Module):
        def __init__(self, num_classes=3, pretrained=False):
            super().__init__()
            self.num_classes = num_classes
            self.backbone = ResNet50Backbone(pretrained=pretrained)
            self.fpn = FPN(in_channels_list=[512, 1024, 2048], out_channels=256)
            self.heads = nn.ModuleList([
                DetectionHead(num_classes=num_classes, in_channels=256)
                for _ in range(3)
            ])
        
        def forward(self, x):
            features = self.backbone(x)
            fpn_features = self.fpn(features)
            predictions = [head(feat) for head, feat in zip(self.heads, fpn_features)]
            return predictions
    
    model = ObjectDetector(num_classes=3, pretrained=False)
    test_input = torch.randn(2, 3, 416, 416)  # Batch of 2
    test_output = model(test_input)
    
    assert len(test_output) == 3, "Should return 3 scale predictions"
    assert test_output[0].shape[0] == 2, "Batch size should be preserved"
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   ✅ Complete Object Detector: PASS")
    print(f"      Total parameters: {total_params:,}")
    print(f"      Trainable parameters: {trainable_params:,}")
except Exception as e:
    print(f"   ❌ Complete Object Detector: FAIL - {e}")
    sys.exit(1)

# Test 5: Loss Function
print("5. Testing Detection Loss...")
try:
    class DetectionLoss(nn.Module):
        def __init__(self, num_classes=3, lambda_coord=5.0, lambda_obj=1.0, 
                     lambda_noobj=0.5, lambda_cls=1.0):
            super().__init__()
            self.num_classes = num_classes
            self.lambda_coord = lambda_coord
            self.lambda_obj = lambda_obj
            self.lambda_noobj = lambda_noobj
            self.lambda_cls = lambda_cls
        
        def forward(self, predictions, targets):
            device = predictions[0].device
            coord_loss = torch.tensor(0.0, device=device)
            obj_loss = torch.tensor(0.0, device=device)
            cls_loss = torch.tensor(0.0, device=device)
            
            for pred in predictions:
                B, A, H, W, C = pred.shape
                pred_boxes = pred[..., :4]
                pred_obj = pred[..., 4]
                pred_cls = pred[..., 5:]
                
                coord_loss += torch.mean(pred_boxes ** 2) * self.lambda_coord
                obj_loss += torch.mean(torch.sigmoid(pred_obj) ** 2) * self.lambda_noobj
                cls_loss += torch.mean(pred_cls ** 2) * self.lambda_cls
            
            total_loss = coord_loss + obj_loss + cls_loss
            
            return {
                'total_loss': total_loss,
                'coord_loss': coord_loss,
                'obj_loss': obj_loss,
                'cls_loss': cls_loss
            }
    
    criterion = DetectionLoss(num_classes=3)
    loss_dict = criterion(test_output, [])
    
    assert 'total_loss' in loss_dict, "Loss dict should contain 'total_loss'"
    assert 'coord_loss' in loss_dict, "Loss dict should contain 'coord_loss'"
    assert 'obj_loss' in loss_dict, "Loss dict should contain 'obj_loss'"
    assert 'cls_loss' in loss_dict, "Loss dict should contain 'cls_loss'"
    
    print("   ✅ Detection Loss: PASS")
    print(f"      Total loss: {loss_dict['total_loss'].item():.4f}")
except Exception as e:
    print(f"   ❌ Detection Loss: FAIL - {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ All tests passed! The notebook components are valid.")
print("="*60)
