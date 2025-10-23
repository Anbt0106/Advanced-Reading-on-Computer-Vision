# How to Use the Object Detection Notebook

## Quick Start Guide

### For Google Colab (Recommended)

1. **Upload the notebook:**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Click "File" > "Upload notebook"
   - Upload `ObjectDetection_Complete.ipynb`

2. **Enable GPU (Optional but Recommended):**
   - Click "Runtime" > "Change runtime type"
   - Select "GPU" from the Hardware accelerator dropdown
   - Click "Save"

3. **Run the notebook:**
   - Click "Runtime" > "Run all"
   - Or press `Ctrl+F9` (Windows/Linux) or `Cmd+F9` (Mac)

4. **Wait for completion:**
   - The notebook will:
     - Install required packages (~1-2 minutes)
     - Create mock datasets (~30 seconds)
     - Train the model (~5-10 minutes on GPU, ~30 minutes on CPU)
     - Generate visualizations and metrics

5. **View results:**
   - Scroll down to see:
     - Dataset samples with bounding boxes
     - Training curves
     - Detection results
     - Confusion matrix
     - Classification report

### For Kaggle

1. **Create a new notebook:**
   - Go to [Kaggle](https://www.kaggle.com/)
   - Click "Code" > "New Notebook"

2. **Import the notebook:**
   - Click "File" > "Import Notebook"
   - Upload `ObjectDetection_Complete.ipynb`

3. **Enable GPU:**
   - Click "Settings" (right sidebar)
   - Accelerator: Select "GPU T4 x2"
   - Internet: Turn ON (for package installation)

4. **Run all cells:**
   - Click "Run All" button

### For Local Jupyter

1. **Install dependencies:**
   ```bash
   pip install torch torchvision numpy pandas matplotlib seaborn \
               scikit-learn tqdm pillow opencv-python
   ```

2. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

3. **Open and run:**
   - Navigate to `ObjectDetection_Complete.ipynb`
   - Run all cells (Cell > Run All)

## Expected Outputs

### During Execution

1. **Setup phase:**
   ```
   ✅ Running on Google Colab
   ✅ Using device: cuda
      GPU: Tesla T4
      Memory: 14.76 GB
   ```

2. **Data preparation:**
   ```
   📥 Downloading COCO sample data...
   📝 Creating mock COCO data (cats and dogs)...
   ✅ Created 100 mock images
      cat: 50 images
      dog: 50 images
   
   🐼 Downloading panda dataset...
   ✅ Created 50 mock panda images
   
   🔄 Combining datasets...
      cat: 50 files
      dog: 50 files
      panda: 50 files
      Total: 150 files
      Train: 120, Val: 30
   ```

3. **Model building:**
   ```
   ✅ ResNet50 Backbone loaded
      Input: torch.Size([1, 3, 416, 416])
      Feature P3: torch.Size([1, 512, 52, 52])
      Feature P4: torch.Size([1, 1024, 26, 26])
      Feature P5: torch.Size([1, 2048, 13, 13])
   
   ✅ FPN Neck loaded
      FPN Feature 3: torch.Size([1, 256, 52, 52])
      FPN Feature 4: torch.Size([1, 256, 26, 26])
      FPN Feature 5: torch.Size([1, 256, 13, 13])
   
   ✅ Complete Object Detector initialized
      Total parameters: 29,757,832
      Trainable parameters: 29,757,832
   ```

4. **Training:**
   ```
   🚀 Starting training...
   
   Epoch 1/5: 100%|██████████| 8/8 [00:12<00:00,  1.50s/it, loss=3.45, avg_loss=3.52]
   Validating: 100%|██████████| 2/2 [00:01<00:00,  1.20it/s]
   
   Epoch 1/5
     Train Loss: 3.5234
     Val Loss: 3.4123
     LR: 0.000951
     ✅ Best model saved! (Val Loss: 3.4123)
   ```

5. **Evaluation:**
   ```
   📊 Evaluating model...
   Evaluating: 100%|██████████| 2/2 [00:01<00:00,  1.50it/s]
   
   ✅ Evaluation complete
      Accuracy: 0.6333
   
   📊 Classification Report:
                 precision    recall  f1-score   support
   
            cat       0.67      0.60      0.63        10
            dog       0.60      0.70      0.65        10
          panda       0.64      0.60      0.62        10
   
       accuracy                           0.63        30
   ```

### Generated Files

After successful execution:

```
Lab5/
├── ObjectDetection_Complete.ipynb
├── best_detector.pth              (New - ~113MB)
├── object_detector_final.pth      (New - ~113MB)
├── dataset_samples.png            (New - visualization)
├── training_curves.png            (New - training plots)
├── detection_result.png           (New - sample detection)
├── confusion_matrix.png           (New - evaluation)
└── yolo_dataset/                  (New - combined dataset)
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/
```

## Customization Options

### Change Training Configuration

Edit the Config class in cell 2:

```python
class Config:
    # Increase training epochs
    EPOCHS = 20  # Default: 5
    
    # Adjust batch size (if memory issues)
    BATCH_SIZE = 8  # Default: 16
    
    # Change learning rate
    LR = 0.0005  # Default: 0.001
    
    # Modify image size
    IMG_SIZE = 640  # Default: 416
```

### Use Real Roboflow Data

Set API key before running:

```python
import os
os.environ['ROBOFLOW_API_KEY'] = 'your_api_key_here'
```

Then uncomment and modify the Roboflow download code in section 3.2.

### Add Your Own Images

1. Place images in appropriate directories:
   ```
   yolo_dataset/train/images/
   yolo_dataset/train/labels/
   yolo_dataset/val/images/
   yolo_dataset/val/labels/
   ```

2. Label format (YOLO):
   ```
   class_id x_center y_center width height
   ```
   All values normalized to [0, 1]

## Troubleshooting

### Problem: CUDA out of memory

**Solution:**
```python
cfg.BATCH_SIZE = 4  # Reduce from 16
```

### Problem: Training too slow

**Solutions:**
- Enable GPU in Colab/Kaggle
- Reduce number of epochs
- Use smaller image size (e.g., 320 instead of 416)

### Problem: No module named 'X'

**Solution:**
Run the first cell to install dependencies:
```python
!pip install -q package_name
```

### Problem: Mock data has low accuracy

**Expected behavior:** Mock data is random, so accuracy will be low (~33-50%).

**Solution:** Use real labeled images for actual training.

## Tips for Best Results

1. **Use GPU:** Training is 10-20x faster on GPU
2. **Increase epochs:** For real training, use 20-50 epochs
3. **Monitor loss:** Loss should decrease over epochs
4. **Check visualizations:** Verify dataset samples look correct
5. **Save models:** Best model is automatically saved
6. **Experiment:** Try different learning rates and batch sizes

## Expected Runtime

| Environment | GPU | Epochs | Estimated Time |
|-------------|-----|--------|----------------|
| Colab       | T4  | 5      | 10-15 minutes  |
| Colab       | CPU | 5      | 30-40 minutes  |
| Kaggle      | P100| 5      | 8-12 minutes   |
| Local       | RTX | 5      | 5-10 minutes   |

## Next Steps

After running the notebook successfully:

1. **Review Results:** Check all visualizations and metrics
2. **Experiment:** Modify hyperparameters and retrain
3. **Real Data:** Replace mock data with actual labeled images
4. **Improve Loss:** Implement proper YOLO loss with IoU
5. **Add NMS:** Implement Non-Maximum Suppression
6. **Deploy:** Export model for inference on new images

## Getting Help

If you encounter issues:

1. Check error messages carefully
2. Review the troubleshooting section above
3. Verify all cells executed in order
4. Check GPU is enabled (if using Colab/Kaggle)
5. Restart runtime and try again

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [YOLO Format Guide](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
- [Roboflow Datasets](https://universe.roboflow.com/)
- [Object Detection Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics)

---

**Enjoy building your object detector! 🚀**
