# voc_to_yolo.py
import os
import xml.etree.ElementTree as ET
from PIL import Image
import shutil
import yaml


class VOCtoYOLO:
    def __init__(self, voc_root, output_root):
        self.voc_root = voc_root
        self.output_root = output_root
        self.class_names = []

    def convert_bbox(self, size, box):
        """Convert VOC bbox to YOLO normalized format"""
        dw = 1.0 / size[0]  # 1/width
        dh = 1.0 / size[1]  # 1/height
        x = (box[0] + box[1]) / 2.0  # center x
        y = (box[2] + box[3]) / 2.0  # center y
        w = box[1] - box[0]  # width
        h = box[3] - box[2]  # height

        # Normalize
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)

    def convert_annotation(self, xml_path):
        """Convert single VOC XML to YOLO format"""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get image dimensions
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        yolo_annotations = []

        for obj in root.findall('object'):
            # Get class name
            class_name = obj.find('name').text
            if class_name not in self.class_names:
                self.class_names.append(class_name)
            class_id = self.class_names.index(class_name)

            # Get bounding box
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # Convert to YOLO format
            yolo_bbox = self.convert_bbox((w, h), (xmin, xmax, ymin, ymax))

            yolo_annotations.append(f"{class_id} {' '.join([f'{x:.6f}' for x in yolo_bbox])}")

        return yolo_annotations

    def process_split(self, file_list, split_name):
        """Process train or val split"""

        images_out = os.path.join(self.output_root, split_name, 'images')
        labels_out = os.path.join(self.output_root, split_name, 'labels')
        os.makedirs(images_out, exist_ok=True)
        os.makedirs(labels_out, exist_ok=True)

        processed = 0

        for filename in file_list:
            # Paths
            xml_path = os.path.join(self.voc_root, 'Annotations', f'{filename}.xml')
            img_path = os.path.join(self.voc_root, 'JPEGImages', f'{filename}.jpg')

            if not (os.path.exists(xml_path) and os.path.exists(img_path)):
                continue

            # Convert annotation
            try:
                yolo_labels = self.convert_annotation(xml_path)

                # Copy image
                dst_img = os.path.join(images_out, f'{filename}.jpg')
                shutil.copy2(img_path, dst_img)

                # Save YOLO label
                dst_label = os.path.join(labels_out, f'{filename}.txt')
                with open(dst_label, 'w') as f:
                    f.write('\n'.join(yolo_labels))

                processed += 1

                if processed % 100 == 0:
                    print(f"   Processed {processed} {split_name} files...")

            except Exception as e:
                print(f"   Error processing {filename}: {e}")
                continue

        print(f"✅ {split_name.capitalize()} set: {processed} files processed")
        return processed

    def convert(self):
        """Main conversion function"""

        # Read splits
        imageset_dir = os.path.join(self.voc_root, 'ImageSets', 'Main')

        with open(os.path.join(imageset_dir, 'train.txt'), 'r') as f:
            train_files = [line.strip() for line in f.readlines()]

        with open(os.path.join(imageset_dir, 'val.txt'), 'r') as f:
            val_files = [line.strip() for line in f.readlines()]

        print(f"🔄 Converting VOC to YOLO format...")
        print(f"   Train files: {len(train_files)}")
        print(f"   Val files: {len(val_files)}")

        # Process splits
        train_count = self.process_split(train_files, 'train')
        val_count = self.process_split(val_files, 'val')

        # Create dataset.yaml
        self.create_dataset_yaml()

        print(f"\n🎉 Conversion completed!")
        print(f"   Train: {train_count} images")
        print(f"   Val: {val_count} images")
        print(f"   Classes: {len(self.class_names)}")
        print(f"   Class names: {self.class_names}")

    def create_dataset_yaml(self):
        """Create dataset.yaml for training"""

        config = {
            'path': os.path.abspath(self.output_root),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(self.class_names),
            'names': {i: name for i, name in enumerate(self.class_names)}
        }

        yaml_path = os.path.join(self.output_root, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"📄 Dataset config saved: {yaml_path}")


def main():
    voc_root = "E:\Pycharm\Advanced-Reading-on-Computer-Vision\Datasets\VOC"  # Thay bằng path VOC của bạn
    output_root = "yolo_dataset"

    converter = VOCtoYOLO(voc_root, output_root)
    converter.convert()


if __name__ == "__main__":
    main()