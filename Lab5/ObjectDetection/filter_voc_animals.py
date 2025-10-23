# filter_voc_animals.py
import os
import xml.etree.ElementTree as ET
import shutil
from collections import Counter


def analyze_animal_annotations(voc_root):
    """Phân tích chi tiết animal class trong VOC"""

    annotations_dir = os.path.join(voc_root, 'Annotations')
    animal_details = Counter()

    animal_files = []

    for xml_file in os.listdir(annotations_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(annotations_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            has_animal = False
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name == 'animal':
                    has_animal = True
                    # Thử extract thêm thông tin nếu có
                    difficult = obj.find('difficult')
                    if difficult is not None:
                        animal_details[f"difficult_{difficult.text}"] += 1
                    else:
                        animal_details["normal"] += 1

            if has_animal:
                animal_files.append(xml_file.replace('.xml', ''))

    print(f"📊 Animal Analysis:")
    print(f"   Files with animals: {len(animal_files)}")
    print(f"   Animal details: {dict(animal_details)}")

    return animal_files


def create_animal_subset(voc_root, output_root, max_samples=500):
    """Tạo subset với animal class từ VOC"""

    # Analyze animals
    animal_files = analyze_animal_annotations(voc_root)

    # Limit samples
    if len(animal_files) > max_samples:
        import random
        random.shuffle(animal_files)
        animal_files = animal_files[:max_samples]

    print(f"Creating animal subset: {len(animal_files)} files")

    # Create output dirs
    os.makedirs(f"{output_root}/images", exist_ok=True)
    os.makedirs(f"{output_root}/labels", exist_ok=True)

    processed = 0

    for filename in animal_files:
        # Paths
        xml_path = os.path.join(voc_root, 'Annotations', f'{filename}.xml')
        img_path = os.path.join(voc_root, 'JPEGImages', f'{filename}.jpg')

        if not (os.path.exists(xml_path) and os.path.exists(img_path)):
            continue

        try:
            # Parse XML
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Get image size
            size = root.find('size')
            img_w = int(size.find('width').text)
            img_h = int(size.find('height').text)

            # Convert animals to YOLO format (class 0 for now)
            yolo_labels = []

            for obj in root.findall('object'):
                if obj.find('name').text == 'animal':
                    bbox = obj.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)

                    # Convert to YOLO normalized format
                    x_center = (xmin + xmax) / 2.0 / img_w
                    y_center = (ymin + ymax) / 2.0 / img_h
                    width = (xmax - xmin) / img_w
                    height = (ymax - ymin) / img_h

                    yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            if yolo_labels:  # Only process if has animals
                # Copy image
                shutil.copy2(img_path, f"{output_root}/images/{filename}.jpg")

                # Save YOLO label
                with open(f"{output_root}/labels/{filename}.txt", 'w') as f:
                    f.write('\n'.join(yolo_labels))

                processed += 1

                if processed % 50 == 0:
                    print(f"   Processed {processed} files...")

        except Exception as e:
            print(f"   Error processing {filename}: {e}")

    print(f"Animal subset created: {processed} files")
    return processed


if __name__ == "__main__":
    voc_root = "E:\Pycharm\Advanced-Reading-on-Computer-Vision\Datasets\VOC"
    output_root = "animal_subset"

    create_animal_subset(voc_root, output_root, max_samples=300)