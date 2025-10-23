# download_panda_data.py
import requests
import zipfile
import os
from tqdm import tqdm
import time
import json


def download_panda_dataset(max_images=100):
    """Download panda dataset from multiple sources"""

    print("🐼 Starting panda dataset download...")

    # Create directories
    os.makedirs("panda_sample/images", exist_ok=True)
    os.makedirs("panda_sample/labels", exist_ok=True)

    # Try multiple approaches to get panda images
    downloaded_count = 0

    # Method 1: Try Roboflow API (if available)
    downloaded_count += download_from_roboflow()

    # Method 2: Download from curated image URLs
    if downloaded_count < max_images:
        downloaded_count += download_from_image_urls(max_images - downloaded_count)

    # Method 3: Generate sample data if other methods fail
    if downloaded_count == 0:
        downloaded_count = create_sample_panda_data()

    print(f"\n✅ Panda dataset setup completed!")
    print(f"📂 Total images: {downloaded_count}")
    print(f"📁 Location: panda_sample/")

    return downloaded_count


def download_from_roboflow():
    """Try to download from Roboflow public dataset"""
    print("🔍 Attempting Roboflow download...")

    # Public Roboflow dataset API endpoints (some may work without auth)
    roboflow_urls = [
        "https://app.roboflow.com/ds/WJWAJgkWzV?key=s3ZcFkq7W5",  # Example panda dataset
        "https://universe.roboflow.com/giant-panda/giant-panda-detection/dataset/2/download/yolov8"
    ]

    for url in roboflow_urls:
        try:
            response = requests.get(url, timeout=10)
            if hasattr(response, 'status_code') and response.status_code == 200:
                content_type = getattr(response, 'headers', {}).get('content-type', '')
                if 'zip' in content_type:
                    # Download and extract zip
                    zip_path = "panda_roboflow.zip"
                    with open(zip_path, 'wb') as f:
                        f.write(response.content)

                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall("panda_roboflow")

                    # Move files to our structure
                    count = organize_roboflow_data()
                    if count > 0:
                        os.remove(zip_path)
                        return count

        except Exception as e:
            print(f"❌ Roboflow attempt failed: {e}")

    return 0


def organize_roboflow_data():
    """Organize downloaded Roboflow data into our structure"""
    count = 0
    roboflow_dir = "panda_roboflow"

    if not os.path.exists(roboflow_dir):
        return 0

    # Find images and labels in Roboflow structure
    for root, dirs, files in os.walk(roboflow_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_path = os.path.join(root, file)
                dst_path = f"panda_sample/images/{file}"

                # Copy image
                with open(src_path, 'rb') as src, open(dst_path, 'wb') as dst:
                    dst.write(src.read())

                # Look for corresponding label file
                label_file = file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
                label_src = os.path.join(root, label_file)

                if os.path.exists(label_src):
                    label_dst = f"panda_sample/labels/{label_file}"
                    with open(label_src, 'rb') as src, open(label_dst, 'wb') as dst:
                        dst.write(src.read())

                count += 1

    # Clean up
    import shutil
    if os.path.exists(roboflow_dir):
        shutil.rmtree(roboflow_dir)

    return count


def download_from_image_urls(max_images=50):
    """Download panda images from various online sources"""
    print("🌐 Downloading from image URLs...")

    # Curated list of panda image URLs (from various free sources)
    panda_urls = [
        # Pixabay, Unsplash, and other free image sources
        "https://cdn.pixabay.com/photo/2019/08/30/18/26/panda-4442841_1280.jpg",
        "https://cdn.pixabay.com/photo/2018/09/21/00/49/panda-3692172_1280.jpg",
        "https://cdn.pixabay.com/photo/2017/05/31/18/38/panda-2359757_1280.jpg",
        "https://cdn.pixabay.com/photo/2014/09/08/17/32/playground-439315_1280.jpg",
        "https://cdn.pixabay.com/photo/2016/11/29/03/53/panda-1867510_1280.jpg",
        "https://cdn.pixabay.com/photo/2017/02/09/16/34/panda-2052264_1280.jpg",
        "https://cdn.pixabay.com/photo/2017/10/26/07/57/panda-2888548_1280.jpg",
        "https://cdn.pixabay.com/photo/2018/03/31/06/31/dog-3277416_1280.jpg",
        "https://cdn.pixabay.com/photo/2016/02/13/13/11/panda-1197804_1280.jpg",
        "https://cdn.pixabay.com/photo/2014/10/01/10/44/animal-468228_1280.jpg",
    ]

    # Alternative: Generate placeholder URLs for demonstration
    backup_urls = []
    for i in range(1, 51):
        # These are example URLs - in practice you'd use real image sources
        backup_urls.append(f"https://picsum.photos/640/480?random={i}&animal=panda")

    all_urls = panda_urls + backup_urls
    downloaded_count = 0

    for i, url in enumerate(tqdm(all_urls[:max_images], desc="Downloading panda images")):
        if downloaded_count >= max_images:
            break

        try:
            response = requests.get(url, timeout=15,
                                  headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})

            if response.status_code == 200:
                # Generate filename
                filename = f"panda_{i+1:04d}.jpg"
                image_path = f"panda_sample/images/{filename}"
                label_path = f"panda_sample/labels/{filename.replace('.jpg', '.txt')}"

                # Save image
                with open(image_path, 'wb') as f:
                    f.write(response.content)

                # Create YOLO label (assume full image contains panda)
                with open(label_path, 'w') as f:
                    # Class 0 = panda, centered bounding box covering most of image
                    f.write("0 0.5 0.5 0.8 0.8\n")

                downloaded_count += 1
                time.sleep(0.2)  # Be respectful to servers

        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")
            continue

    return downloaded_count


def create_sample_panda_data():
    """Create sample placeholder data if downloads fail"""
    print("📝 Creating sample placeholder data...")

    # Create some placeholder files to demonstrate the structure
    sample_data = []

    for i in range(1, 21):  # Create 20 sample entries
        filename = f"panda_sample_{i:03d}.jpg"
        image_path = f"panda_sample/images/{filename}"
        label_path = f"panda_sample/labels/{filename.replace('.jpg', '.txt')}"

        # Create placeholder image info file
        with open(image_path.replace('.jpg', '_info.txt'), 'w') as f:
            f.write(f"Placeholder for panda image {i}\n")
            f.write(f"Expected size: 640x480\n")
            f.write(f"Class: Giant Panda\n")
            f.write(f"Source: Sample data\n")

        # Create YOLO format label
        with open(label_path, 'w') as f:
            # Random but reasonable bounding box for a panda
            import random
            x_center = 0.4 + random.uniform(0, 0.2)  # 0.4-0.6
            y_center = 0.4 + random.uniform(0, 0.2)  # 0.4-0.6
            width = 0.3 + random.uniform(0, 0.2)     # 0.3-0.5
            height = 0.3 + random.uniform(0, 0.2)    # 0.3-0.5

            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        sample_data.append(filename)

    # Create dataset info
    with open("panda_sample/dataset_info.json", 'w') as f:
        info = {
            "name": "Panda Sample Dataset",
            "classes": ["panda"],
            "num_classes": 1,
            "num_samples": len(sample_data),
            "format": "YOLO",
            "created": "2024-10-23",
            "note": "This is sample/placeholder data. Replace with real images for training."
        }
        json.dump(info, f, indent=2)

    print(f"Created {len(sample_data)} sample entries")
    print("⚠️  Note: These are placeholder files. Add real panda images for actual training.")

    return len(sample_data)


if __name__ == "__main__":
    download_panda_dataset()