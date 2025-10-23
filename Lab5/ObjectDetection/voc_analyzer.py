# voc_analyzer.py
import os
import xml.etree.ElementTree as ET
from collections import Counter
import matplotlib.pyplot as plt


def analyze_voc_dataset(voc_root):
    """Phân tích dataset VOC có sẵn"""

    annotations_dir = os.path.join(voc_root, 'Annotations')
    imageset_dir = os.path.join(voc_root, 'ImageSets', 'Main')
    images_dir = os.path.join(voc_root, 'JPEGImages')

    # Đọc train/val splits
    train_files = []
    val_files = []

    if os.path.exists(os.path.join(imageset_dir, 'train.txt')):
        with open(os.path.join(imageset_dir, 'train.txt'), 'r') as f:
            train_files = [line.strip() for line in f.readlines()]

    if os.path.exists(os.path.join(imageset_dir, 'val.txt')):
        with open(os.path.join(imageset_dir, 'val.txt'), 'r') as f:
            val_files = [line.strip() for line in f.readlines()]

    print(f"📊 VOC Dataset Analysis:")
    print(f"   Train files: {len(train_files)}")
    print(f"   Val files: {len(val_files)}")

    # Analyze annotations
    all_classes = Counter()
    total_objects = 0

    for xml_file in os.listdir(annotations_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(annotations_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for obj in root.findall('object'):
                class_name = obj.find('name').text
                all_classes[class_name] += 1
                total_objects += 1

    print(f"   Total objects: {total_objects}")
    print(f"   Classes found: {len(all_classes)}")
    print(f"   Class distribution:")
    for class_name, count in all_classes.most_common():
        print(f"     {class_name}: {count}")

    return train_files, val_files, list(all_classes.keys())


if __name__ == "__main__":
    voc_root = "E:\Pycharm\Advanced-Reading-on-Computer-Vision\Datasets\VOC"  # Path đến thư mục VOC của bạn
    analyze_voc_dataset(voc_root)