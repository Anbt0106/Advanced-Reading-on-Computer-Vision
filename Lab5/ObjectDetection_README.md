# Object Detection Project - Lab 5

## Overview

This is a complete Object Detection project implementing a **Backbone + Neck + Head** architecture for detecting 3 classes: **Cat**, **Dog**, and **Panda**.

## Architecture

- **Backbone**: ResNet50 (pretrained on ImageNet) - Feature extraction
- **Neck**: Feature Pyramid Network (FPN) - Multi-scale feature fusion  
- **Head**: Detection heads - Classification + Regression + Objectness

## Dataset

- **COCO Dataset**: Cats and Dogs
- **Roboflow/Mock Data**: Pandas (with fallback for missing API key)
- **Format**: YOLO (class_id x_center y_center width height)
- **Classes**: 
  - 0: cat
  - 1: dog
  - 2: panda

## Files

- **ObjectDetection_Complete.ipynb**: Main notebook with complete implementation
- **models/**: Model components (FPN, YOLO head, etc.)
- **utils/**: Utility functions (dataset, loss, etc.)
- **download_coco_cats_dogs.py**: COCO data download script
- **download_panda_data.py**: Panda data download script with Roboflow API
- **combine_detection_dataset.py**: Dataset combination utility

## Quick Start

### Option 1: Google Colab (Recommended)

1. Upload `ObjectDetection_Complete.ipynb` to Google Colab
2. Run all cells sequentially
3. The notebook will:
   - Install required packages
   - Create mock data (or download real data if API keys available)
   - Train the detection model
   - Generate visualizations and metrics

### Option 2: Local/Kaggle

```bash
# Install dependencies
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn tqdm pillow opencv-python

# Open the notebook
jupyter notebook ObjectDetection_Complete.ipynb
```

## Notebook Structure

1. **Setup and Environment** - Package installation and imports
2. **Configuration** - Hyperparameters and paths
3. **Data Pipeline**
   - Download COCO cats/dogs
   - Download panda data (with Roboflow API or fallback)
   - Combine datasets and create train/val split
   - Data visualization
4. **Model Architecture**
   - ResNet50 Backbone
   - FPN Neck
   - Detection Head
   - Complete Object Detector
5. **Custom Dataset and DataLoader**
6. **Loss Function** - Simplified detection loss
7. **Training Pipeline**
   - Training loop
   - Validation
   - Training curves visualization
8. **Inference and Evaluation**
   - Load best model
   - Inference function
   - Visualization of detections
   - Evaluation metrics (accuracy, confusion matrix, classification report)
9. **Model Saving and Loading**
10. **Summary and Conclusion**

## Training Configuration

- **Image Size**: 416x416
- **Batch Size**: 16
- **Epochs**: 5 (demo configuration)
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Scheduler**: Cosine Annealing

## Output Files

After running the notebook, the following files will be generated:

- `best_detector.pth` - Best model checkpoint
- `object_detector_final.pth` - Final model with configuration
- `dataset_samples.png` - Dataset visualization
- `training_curves.png` - Training progress
- `detection_result.png` - Sample detection visualization
- `confusion_matrix.png` - Evaluation metrics

## Features

✅ **Production-Ready**: Can run on Kaggle/Colab without modifications  
✅ **Modular Design**: Clean separation of Backbone, Neck, and Head  
✅ **Error Handling**: Fallback mock data if downloads fail  
✅ **Comprehensive Visualization**: Training curves, detection results, metrics  
✅ **Full Evaluation**: Confusion matrix, classification report, accuracy  

## API Keys (Optional)

For downloading real panda data from Roboflow:

```python
# Set environment variable before running notebook
import os
os.environ['ROBOFLOW_API_KEY'] = 'your_api_key_here'
```

**Note**: The notebook will use mock data if no API key is provided, so it works out of the box.

## Model Performance

The model uses a simplified loss function for demonstration purposes. For production use:

1. Implement proper YOLO loss with anchor matching
2. Add Non-Maximum Suppression (NMS) post-processing
3. Use mAP (mean Average Precision) metric
4. Train with real labeled images
5. Increase training epochs (20-50 recommended)

## Technical Details

### Input
- RGB images resized to 416x416
- Normalized with ImageNet statistics

### Output
- Multi-scale predictions at 3 levels (P3, P4, P5)
- Each prediction contains: [bbox_coords(4), objectness(1), class_scores(3)]

### Data Augmentation
- Random horizontal flip
- Color jitter
- Resize to fixed size

## Future Improvements

1. **Enhanced Loss**: Implement full YOLO loss with IoU calculation
2. **Anchor Boxes**: Use proper anchor box assignment
3. **NMS**: Add Non-Maximum Suppression for overlapping detections
4. **Data Augmentation**: More sophisticated augmentations (rotation, scaling, mosaic)
5. **mAP Metric**: Implement mean Average Precision evaluation
6. **Real Dataset**: Use actual labeled images from COCO and Roboflow

## Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce batch size in the Config class

```python
cfg.BATCH_SIZE = 8  # or 4
```

### Issue: Dataset not found
**Solution**: Ensure the data pipeline cells are executed before training

### Issue: Slow training
**Solution**: Enable GPU in Colab (Runtime > Change runtime type > GPU)

## References

- ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- FPN: [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)
- YOLO: [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

## License

This project is for educational purposes as part of the Advanced Reading on Computer Vision course.

## Author

**Student ID**: 22001534  
**Course**: Advanced Reading on Computer Vision  
**Lab**: Lab 5 - Object Detection
