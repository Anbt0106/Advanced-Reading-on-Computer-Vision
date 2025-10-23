# utils/dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


class DetectionDataset(Dataset):
    def __init__(self, data_root, split='train', img_size=416, augment=False):
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.augment = augment

        # Paths
        self.images_dir = os.path.join(data_root, split, 'images')
        self.labels_dir = os.path.join(data_root, split, 'labels')

        # Get image files
        self.image_files = [f for f in os.listdir(self.images_dir)
                            if f.endswith(('.jpg', '.jpeg', '.png'))]

        # Transforms
        self.setup_transforms()

        print(f"📁 {split.capitalize()} dataset: {len(self.image_files)} images")

    def setup_transforms(self):
        """Setup image transforms"""
        transforms_list = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ]

        if self.augment and self.split == 'train':
            transforms_list.insert(-1, transforms.ColorJitter(0.1, 0.1, 0.1, 0.1))
            transforms_list.insert(-1, transforms.RandomHorizontalFlip(0.5))

        self.transform = transforms.Compose(transforms_list)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Load labels
        label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(self.labels_dir, label_name)

        targets = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        targets.append([class_id, x, y, w, h])

        # Convert to tensor
        targets = torch.tensor(targets, dtype=torch.float32)

        # Apply transforms
        original_size = image.size
        image = self.transform(image)

        # Adjust targets if image was resized (already normalized, so no change needed)

        return image, targets