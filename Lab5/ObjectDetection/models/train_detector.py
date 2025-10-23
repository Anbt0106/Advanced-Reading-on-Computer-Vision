# train_detector.py - Updated version with proper imports
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import time
from pathlib import Path

# Import models và utils
from Lab5.ObjectDetection.models.eelan_detector import EELANDetector
from Lab5.ObjectDetection.utils.dataset import DetectionDataset
from Lab5.ObjectDetection.utils.loss import YOLOLoss
from Lab5.ObjectDetection.utils.metrics import SimplifiedEvaluator  # Dùng simplified version


class DetectionTrainer:
    def __init__(self, config_path):
        # Load config
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Setup directories
        self.output_dir = Path(self.cfg['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset info
        with open(self.cfg['data']['yaml_path'], 'r') as f:
            self.data_cfg = yaml.safe_load(f)

        self.num_classes = self.data_cfg['nc']
        self.class_names = list(self.data_cfg['names'].values())

        print(f"Dataset: {self.num_classes} classes - {self.class_names}")

        # Initialize evaluator
        self.evaluator = SimplifiedEvaluator(self.num_classes, self.class_names)

        # Initialize model
        self.setup_model()

        # Initialize loss
        self.criterion = YOLOLoss(
            num_classes=self.num_classes,
            lambda_coord=self.cfg['loss']['lambda_coord'],
            lambda_noobj=self.cfg['loss']['lambda_noobj']
        )

        # Setup optimizer
        self.setup_optimizer()

        # Setup data loaders
        self.setup_data_loaders()

        # Training history
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'map_50': [],
            'map_75': []
        }

        print("Trainer initialized successfully!")

    def setup_model(self):
        """Initialize and setup model"""
        self.model = EELANDetector(
            num_classes=self.num_classes,
            C_stem=self.cfg['model']['C_stem'],
            C_stage=self.cfg['model']['C_stage']
        ).to(self.device)

        # Load pretrained backbone if available
        pretrained = self.cfg['model'].get('pretrained_backbone')
        if pretrained and pretrained != 'null' and os.path.exists(pretrained):
            try:
                self.model.load_classification_backbone(pretrained)
                print("Loaded pretrained classification backbone")
            except Exception as e:
                print(f"Could not load pretrained backbone: {e}")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model: {total_params:,} total params, {trainable_params:,} trainable")

    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg['training']['lr'],
            weight_decay=self.cfg['training']['weight_decay']
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg['training']['epochs'],
            eta_min=self.cfg['training']['lr'] * 0.01
        )

    def setup_data_loaders(self):
        """Setup training and validation data loaders"""

        try:
            # Training dataset
            train_dataset = DetectionDataset(
                data_root=self.data_cfg['path'],
                split='train',
                img_size=self.cfg['training']['img_size'],
                augment=True
            )

            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.cfg['training']['batch_size'],
                shuffle=True,
                num_workers=self.cfg['training']['num_workers'],
                collate_fn=self.collate_fn,
                pin_memory=True if self.device.type == 'cuda' else False
            )

            # Validation dataset
            val_dataset = DetectionDataset(
                data_root=self.data_cfg['path'],
                split='val',
                img_size=self.cfg['training']['img_size'],
                augment=False
            )

            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.cfg['training']['batch_size'],
                shuffle=False,
                num_workers=self.cfg['training']['num_workers'],
                collate_fn=self.collate_fn,
                pin_memory=True if self.device.type == 'cuda' else False
            )

            print(f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} val")

        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for detection"""
        images = []
        targets = []

        for i, (img, target) in enumerate(batch):
            images.append(img)
            # Add batch index to targets
            if target.size(0) > 0:
                batch_idx = torch.full((target.size(0), 1), i, dtype=target.dtype)
                target = torch.cat([batch_idx, target], dim=1)
            targets.append(target)

        images = torch.stack(images, 0)
        targets = torch.cat(targets, 0) if len([t for t in targets if t.size(0) > 0]) > 0 else torch.empty(0, 6)

        return images, targets

    def train_one_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Forward pass
            predictions = self.model(images)

            # Compute loss
            loss_dict = self.criterion(
                predictions, targets,
                self.model.anchors.to(self.device),
                self.model.strides.to(self.device)
            )

            loss = loss_dict['total_loss']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.cfg['training']['grad_clip']
            )

            self.optimizer.step()

            total_loss += loss.item()

            # Print progress
            if batch_idx % self.cfg['training']['print_freq'] == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:3d}/{self.cfg['training']['epochs']:3d} "
                      f"[{batch_idx:4d}/{num_batches:4d}] "
                      f"Loss: {loss.item():.4f} "
                      f"LR: {current_lr:.6f}")

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        self.evaluator.reset()

        for images, targets in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Forward pass
            predictions = self.model(images)

            # Compute loss
            loss_dict = self.criterion(
                predictions, targets,
                self.model.anchors.to(self.device),
                self.model.strides.to(self.device)
            )

            total_loss += loss_dict['total_loss'].item()

            # Update evaluator
            self.evaluator.update(predictions, targets)

        avg_loss = total_loss / len(self.val_loader)

        # Calculate mAP (simplified)
        map_50 = self.evaluator.compute_map(iou_thresh=0.5)
        map_75 = self.evaluator.compute_map(iou_thresh=0.75)

        return avg_loss, map_50, map_75

    # ... rest of the methods remain the same ...

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.cfg,
            'class_names': self.class_names,
            'history': self.history
        }

        # Save latest
        latest_path = self.output_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = self.output_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")

    def train(self):
        """Main training loop"""
        print("Starting training...")

        best_map = 0.0
        start_time = time.time()

        try:
            for epoch in range(1, self.cfg['training']['epochs'] + 1):
                epoch_start = time.time()

                # Training
                train_loss = self.train_one_epoch(epoch)

                # Validation
                val_loss, map_50, map_75 = self.validate(epoch)

                # Update learning rate
                self.scheduler.step()

                # Update history
                self.history['epoch'].append(epoch)
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['map_50'].append(map_50)
                self.history['map_75'].append(map_75)

                # Check if best model
                is_best = map_50 > best_map
                if is_best:
                    best_map = map_50

                # Save checkpoint
                self.save_checkpoint(epoch, is_best)

                # Print epoch summary
                epoch_time = time.time() - epoch_start
                print(f"Epoch {epoch:3d} Summary: "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"mAP@0.5: {map_50:.4f}, "
                      f"Time: {epoch_time:.1f}s")

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"\nTraining error: {e}")
            raise

        total_time = time.time() - start_time
        print(f"\nTraining completed!")
        print(f"   Total time: {total_time / 3600:.2f} hours")
        print(f"   Best mAP@0.5: {best_map:.4f}")


def main():
    # Try to find config file in multiple locations
    config_paths = [
        "configs/train_config.yaml",
        "../configs/train_config.yaml",
        "Lab5/ObjectDetection/configs/train_config.yaml"
    ]

    config_path = None
    for path in config_paths:
        if os.path.exists(path):
            config_path = path
            break

    if config_path is None:
        print("Config file not found in any of these locations:")
        for path in config_paths:
            print(f"  - {path}")
        print("\nCreating default config file...")

        # Create default config
        config_path = "configs/train_config.yaml"
        create_default_config(config_path)
        print(f"Created default config: {config_path}")
        print("You may want to modify the config file for your specific needs.")

    try:
        trainer = DetectionTrainer(config_path)
        trainer.train()
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


def create_default_config(config_path):
    """Create a default training configuration file"""
    import os

    # Ensure configs directory exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    default_config = {
        'data': {
            'yaml_path': 'coco_sample/data.yaml',
            'train_images': 'coco_sample/train/images',
            'train_labels': 'coco_sample/train/labels',
            'val_images': 'coco_sample/val/images',
            'val_labels': 'coco_sample/val/labels'
        },
        'model': {
            'C_stem': 32,
            'C_stage': [64, 128, 256, 512],
            'pretrained_backbone': None
        },
        'training': {
            'epochs': 100,
            'batch_size': 8,
            'img_size': 640,
            'lr': 0.001,
            'weight_decay': 0.0005,
            'grad_clip': 10.0,
            'num_workers': 4,
            'print_freq': 10
        },
        'loss': {
            'lambda_coord': 5.0,
            'lambda_noobj': 0.5
        },
        'output_dir': 'runs/train'
    }

    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)


if __name__ == "__main__":
    main()