# combine_detection_dataset.py
import os
import shutil
import yaml
from sklearn.model_selection import train_test_split
import random


def combine_three_class_dataset():
    """Combine cat, dog, panda thành final dataset"""

    # Định nghĩa classes theo đề bài
    classes = {
        0: 'cat',
        1: 'dog',
        2: 'panda'
    }

    # Paths to source data
    sources = {
        'cat': 'E:\\Pycharm\\Advanced-Reading-on-Computer-Vision\\Lab5\\ObjectDetection\\coco_sample\\cat',  # từ COCO
        'dog': 'E:\\Pycharm\\Advanced-Reading-on-Computer-Vision\\Lab5\\ObjectDetection\\coco_sample\\dog',  # từ COCO
        'panda': 'E:\\Pycharm\\Advanced-Reading-on-Computer-Vision\\Lab5\\ObjectDetection\\panda_sample'  # từ Roboflow
    }

    print("Combining three-class dataset...")

    # Collect all files
    all_files = []

    for class_id, class_name in classes.items():
        source_dir = sources[class_name]
        images_dir = os.path.join(source_dir, 'images')
        labels_dir = os.path.join(source_dir, 'labels')

        class_files = 0
        if os.path.exists(images_dir):
            # Get image files
            img_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

            for img_file in img_files:
                # Try multiple label extensions
                label_file = None
                for ext in ['.txt']:
                    potential_label = img_file.rsplit('.', 1)[0] + ext
                    if os.path.exists(os.path.join(labels_dir, potential_label)):
                        label_file = potential_label
                        break

                if label_file:
                    img_path = os.path.join(images_dir, img_file)
                    label_path = os.path.join(labels_dir, label_file)

                    all_files.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'img_path': img_path,
                        'label_path': label_path,
                        'filename': img_file
                    })
                    class_files += 1

            print(f"   {class_name}: {class_files} valid image-label pairs found")
        else:
            print(f"   {class_name}: Directory {images_dir} not found")

    print(f"   Total files: {len(all_files)}")

    # Check if we have any data
    if len(all_files) == 0:
        print("\n❌ ERROR: No valid image-label pairs found!")
        print("Please ensure the source directories contain data:")
        for class_name, source_dir in sources.items():
            print(f"   - {class_name}: {source_dir}")
            images_dir = os.path.join(source_dir, 'images')
            labels_dir = os.path.join(source_dir, 'labels')

            if os.path.exists(images_dir):
                img_count = len([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
                print(f"     Images: {img_count}")
            else:
                print(f"     Images directory missing: {images_dir}")

            if os.path.exists(labels_dir):
                label_count = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
                print(f"     Labels: {label_count}")
            else:
                print(f"     Labels directory missing: {labels_dir}")

        print("\nTo fix this issue:")
        print("1. Run download_coco_cats_dogs.py to get cat/dog data")
        print("2. Run download_panda_data.py to get panda data")
        print("3. Or manually add images and labels to the directories above")
        return None

    # Train/val split
    random.shuffle(all_files)
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

    print(f"   Train: {len(train_files)}, Val: {len(val_files)}")

    # Create output structure
    output_root = "yolo_dataset"
    for split in ['train', 'val']:
        os.makedirs(f"{output_root}/{split}/images", exist_ok=True)
        os.makedirs(f"{output_root}/{split}/labels", exist_ok=True)

    # Copy files
    for split_name, files in [('train', train_files), ('val', val_files)]:
        for i, file_info in enumerate(files):
            # Generate new filename
            new_filename = f"{file_info['class_name']}_{i:04d}.jpg"
            new_label = f"{file_info['class_name']}_{i:04d}.txt"

            # Copy image
            shutil.copy2(
                file_info['img_path'],
                f"{output_root}/{split_name}/images/{new_filename}"
            )

            # Update and copy label (fix class ID)
            with open(file_info['label_path'], 'r') as f:
                labels = f.readlines()

            updated_labels = []
            for label in labels:
                parts = label.strip().split()
                if len(parts) >= 5:
                    # Update class ID
                    parts[0] = str(file_info['class_id'])
                    updated_labels.append(' '.join(parts))

            with open(f"{output_root}/{split_name}/labels/{new_label}", 'w') as f:
                f.write('\n'.join(updated_labels))

    # Create dataset.yaml
    config = {
        'path': os.path.abspath(output_root),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 3,
        'names': classes
    }

    with open(f"{output_root}/dataset.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Final dataset created: {output_root}")
    print(f"   Classes: {list(classes.values())}")

    return output_root


if __name__ == "__main__":
    combine_three_class_dataset()