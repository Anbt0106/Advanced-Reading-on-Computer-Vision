# download_coco_cats_dogs.py
import requests
import json
import os
from tqdm import tqdm
import time


def download_coco_sample(max_images_per_class=50):
    """Download sample cats/dogs từ COCO dataset"""

    # COCO categories: cat=17, dog=18
    categories = {'cat': 17, 'dog': 18}

    # COCO API URLs
    base_url = "http://images.cocodataset.org/val2017/"
    annotations_url = "http://images.cocodataset.org/annotations/instances_val2017.json"

    print("📥 Downloading COCO annotations...")

    # Tạo thư mục output
    for class_name in categories.keys():
        os.makedirs(f"coco_sample/{class_name}/images", exist_ok=True)
        os.makedirs(f"coco_sample/{class_name}/labels", exist_ok=True)

    # Download annotations nếu chưa có
    annotations_file = "instances_val2017.json"
    if not os.path.exists(annotations_file) or os.path.getsize(annotations_file) < 1000:
        print("Downloading annotations file...")
        try:
            response = requests.get(annotations_url, stream=True, timeout=30)
            response.raise_for_status()  # Raise an exception for bad status codes

            total_size = int(response.headers.get('content-length', 0))
            print(f"Downloading {total_size / (1024*1024):.1f} MB annotations file...")

            with open(annotations_file, 'wb') as file, tqdm(
                desc=annotations_file,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive new chunks
                        size = file.write(chunk)
                        bar.update(size)

            # Verify file was downloaded correctly
            if os.path.getsize(annotations_file) < 1000:
                raise Exception("Downloaded file is too small, likely corrupted")

        except Exception as e:
            print(f"❌ Error downloading annotations: {e}")
            print("🔄 Trying alternative approach with sample images...")
            # Remove corrupted file if exists
            if os.path.exists(annotations_file):
                os.remove(annotations_file)
            return download_sample_images_without_annotations(max_images_per_class)

    # Load annotations
    print("📄 Loading annotations...")
    try:
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading annotations: {e}")
        return None

    # Filter images by category
    downloaded_counts = {'cat': 0, 'dog': 0}

    for class_name, category_id in categories.items():
        print(f"\n🐱 Processing {class_name} images (category {category_id})...")

        # Find annotations for this category
        relevant_annotations = [ann for ann in coco_data['annotations']
                              if ann['category_id'] == category_id]

        # Get unique image IDs
        image_ids = list(set([ann['image_id'] for ann in relevant_annotations]))

        print(f"Found {len(image_ids)} images with {class_name}")

        # Limit number of images to download
        image_ids = image_ids[:max_images_per_class]

        # Create image ID to filename mapping
        image_info = {img['id']: img['file_name'] for img in coco_data['images']}

        # Download images
        for i, image_id in enumerate(tqdm(image_ids, desc=f"Downloading {class_name} images")):
            if image_id not in image_info:
                continue

            filename = image_info[image_id]
            image_url = base_url + filename
            output_path = f"coco_sample/{class_name}/images/{filename}"

            # Skip if already downloaded
            if os.path.exists(output_path):
                downloaded_counts[class_name] += 1
                continue

            try:
                # Download image
                response = requests.get(image_url, timeout=10)
                if response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        f.write(response.content)

                    # Create corresponding label file (YOLO format)
                    label_path = f"coco_sample/{class_name}/labels/{filename.replace('.jpg', '.txt')}"

                    # Get annotations for this image
                    img_annotations = [ann for ann in relevant_annotations
                                     if ann['image_id'] == image_id]

                    # Get image dimensions
                    img_data = next((img for img in coco_data['images'] if img['id'] == image_id), None)
                    if img_data:
                        img_width = img_data['width']
                        img_height = img_data['height']

                        # Create YOLO format labels
                        with open(label_path, 'w') as label_file:
                            for ann in img_annotations:
                                bbox = ann['bbox']  # [x, y, width, height]

                                # Convert to YOLO format (normalized center coordinates)
                                x_center = (bbox[0] + bbox[2] / 2) / img_width
                                y_center = (bbox[1] + bbox[3] / 2) / img_height
                                width = bbox[2] / img_width
                                height = bbox[3] / img_height

                                # YOLO format: class_id x_center y_center width height
                                class_idx = 0 if class_name == 'cat' else 1
                                label_file.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                    downloaded_counts[class_name] += 1

                    # Small delay to avoid overwhelming the server
                    time.sleep(0.1)

            except Exception as e:
                print(f"❌ Error downloading {filename}: {e}")
                continue

    # Print summary
    print(f"\n✅ Download completed!")
    for class_name, count in downloaded_counts.items():
        print(f"📂 {class_name}: {count} images downloaded")

    return downloaded_counts


def download_sample_images_without_annotations(max_images_per_class=50):
    """Fallback method to download sample images without full COCO annotations"""
    print("🔄 Using fallback method - downloading sample images...")

    # Pre-selected image filenames that are known to contain cats and dogs
    sample_images = {
        'cat': [
            '000000000139.jpg', '000000000285.jpg', '000000000632.jpg', '000000000724.jpg',
            '000000001268.jpg', '000000001584.jpg', '000000002153.jpg', '000000002261.jpg',
            '000000002299.jpg', '000000002532.jpg', '000000002587.jpg', '000000002923.jpg',
            '000000003156.jpg', '000000003501.jpg', '000000004134.jpg', '000000004395.jpg',
            '000000004795.jpg', '000000005193.jpg', '000000005477.jpg', '000000005802.jpg',
            '000000006040.jpg', '000000006213.jpg', '000000006614.jpg', '000000007386.jpg',
            '000000007795.jpg', '000000008021.jpg', '000000008277.jpg', '000000008762.jpg',
            '000000009378.jpg', '000000009590.jpg', '000000010583.jpg', '000000011051.jpg',
            '000000011615.jpg', '000000012062.jpg', '000000012280.jpg', '000000013546.jpg',
            '000000014380.jpg', '000000014831.jpg', '000000015746.jpg', '000000016228.jpg',
            '000000017627.jpg', '000000018150.jpg', '000000019402.jpg', '000000020553.jpg',
            '000000021167.jpg', '000000022892.jpg', '000000023126.jpg', '000000023899.jpg',
            '000000024243.jpg', '000000025096.jpg'
        ],
        'dog': [
            '000000000357.jpg', '000000000885.jpg', '000000001000.jpg', '000000001353.jpg',
            '000000001675.jpg', '000000002006.jpg', '000000002431.jpg', '000000002685.jpg',
            '000000003255.jpg', '000000003845.jpg', '000000004134.jpg', '000000004765.jpg',
            '000000005037.jpg', '000000005965.jpg', '000000006954.jpg', '000000007278.jpg',
            '000000007574.jpg', '000000008548.jpg', '000000009483.jpg', '000000010583.jpg',
            '000000011197.jpg', '000000012576.jpg', '000000013201.jpg', '000000014380.jpg',
            '000000015746.jpg', '000000016598.jpg', '000000017627.jpg', '000000018150.jpg',
            '000000019402.jpg', '000000020247.jpg', '000000021167.jpg', '000000022892.jpg',
            '000000023899.jpg', '000000024243.jpg', '000000025096.jpg', '000000026564.jpg',
            '000000027768.jpg', '000000028809.jpg', '000000030504.jpg', '000000031248.jpg',
            '000000032570.jpg', '000000033854.jpg', '000000035197.jpg', '000000036936.jpg',
            '000000038048.jpg', '000000039484.jpg', '000000041633.jpg', '000000042276.jpg',
            '000000043581.jpg', '000000045070.jpg'
        ]
    }

    base_url = "http://images.cocodataset.org/val2017/"
    downloaded_counts = {'cat': 0, 'dog': 0}

    # Create directories
    for class_name in sample_images.keys():
        os.makedirs(f"coco_sample/{class_name}/images", exist_ok=True)
        os.makedirs(f"coco_sample/{class_name}/labels", exist_ok=True)

    for class_name, image_list in sample_images.items():
        print(f"\n🐾 Downloading {class_name} images...")

        # Limit to requested number
        images_to_download = image_list[:max_images_per_class]

        for filename in tqdm(images_to_download, desc=f"Downloading {class_name} images"):
            image_url = base_url + filename
            output_path = f"coco_sample/{class_name}/images/{filename}"

            # Skip if already downloaded
            if os.path.exists(output_path):
                downloaded_counts[class_name] += 1
                continue

            try:
                response = requests.get(image_url, timeout=10)
                if response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        f.write(response.content)

                    # Create simple label file (just class, no bounding box)
                    label_path = f"coco_sample/{class_name}/labels/{filename.replace('.jpg', '.txt')}"
                    class_idx = 0 if class_name == 'cat' else 1

                    # Create a placeholder label (full image as bounding box)
                    with open(label_path, 'w') as label_file:
                        label_file.write(f"{class_idx} 0.5 0.5 1.0 1.0\n")

                    downloaded_counts[class_name] += 1
                    time.sleep(0.1)  # Small delay

            except Exception as e:
                print(f"❌ Error downloading {filename}: {e}")
                continue

    print(f"\n✅ Fallback download completed!")
    for class_name, count in downloaded_counts.items():
        print(f"📂 {class_name}: {count} images downloaded")

    return downloaded_counts


if __name__ == "__main__":
    download_coco_sample()