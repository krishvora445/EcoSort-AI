# 🚀 Quick Start Guide
## AI Waste Classification System

This guide will help you get started with the waste classification system in **under 10 minutes**!

---

## ⚡ Quick Setup (5 Steps)

### Step 1: Install Python Dependencies (2 min)

```bash
# Install required packages
pip install tensorflow opencv-python pillow numpy pandas matplotlib seaborn streamlit
```

### Step 2: Download Dataset (3 min)

**Option A - Automatic (Recommended):**
```bash
python download_data.py
```

**Option B - Manual:**
1. Go to: https://www.kaggle.com/datasets/techsash/waste-classification-data
2. Download and extract to `data/DATASET/`

### Step 3: Train Model (30-45 min with GPU)

```bash
python train.py
```

*Note: Training will take 2-3 hours on CPU, 30-45 minutes on GPU*

### Step 4: Test Predictions (30 sec)

```bash
python predict.py --mode image --image path/to/test/image.jpg
```

### Step 5: Launch Web App (30 sec)

```bash
streamlit run app.py
```

Visit: http://localhost:8501

---

## 🎯 Common Use Cases

### Use Case 1: Test on Single Image

```bash
python predict.py --mode image --image samples/plastic_bottle.jpg
```

**Expected Output:**
```
Prediction: Recyclable
Confidence: 96.7%
```

### Use Case 2: Real-time Classification with Webcam

```bash
python predict.py --mode webcam
```

**Controls:**
- `Q` - Quit
- `S` - Save screenshot

### Use Case 3: Batch Processing

```bash
python predict.py --mode batch --dir test_images/
```

**Expected Output:**
```
✓ image1.jpg -> Organic (94.2%)
✓ image2.jpg -> Recyclable (97.1%)
✓ image3.jpg -> Organic (89.3%)
...
```

### Use Case 4: Web Interface

```bash
streamlit run app.py
```

**Features:**
- Upload images via drag-and-drop
- View confidence scores
- Get disposal recommendations
- Interactive visualizations

---

## 🐛 Troubleshooting

### Problem 1: Import Error

```
ModuleNotFoundError: No module named 'tensorflow'
```

**Solution:**
```bash
pip install tensorflow keras opencv-python numpy pandas matplotlib seaborn
```

### Problem 2: Dataset Not Found

```
FileNotFoundError: [Errno 2] No such file or directory: 'data/DATASET'
```

**Solution:**
```bash
# Make sure dataset is downloaded
python download_data.py

# Or manually place dataset in correct location:
# data/DATASET/O/  (Organic)
# data/DATASET/R/  (Recyclable)
```

### Problem 3: CUDA Out of Memory (GPU)

```
tensorflow.python.framework.errors_impl.ResourceExhaustedError
```

**Solution:**
```python
# Reduce batch size in train.py
BATCH_SIZE = 16  # Change from 32 to 16
```

### Problem 4: Webcam Not Working

```
Error: Could not open webcam
```

**Solution:**
```bash
# Check camera permissions
# Try different camera index:
python predict.py --mode webcam --camera 1
```

---

## 📊 Expected Results

### Training Metrics (After 25 Epochs)

| Metric | Value |
|--------|-------|
| Training Accuracy | 97.2% |
| Validation Accuracy | 95.3% |
| Training Loss | 0.081 |
| Validation Loss | 0.142 |

### Prediction Speed

| Hardware | Inference Time |
|----------|---------------|
| CPU (Intel i5) | ~200ms |
| GPU (NVIDIA GTX 1060) | ~50ms |
| Raspberry Pi 4 | ~500ms |

---

## 🎓 Learning Path

### For Beginners

1. **Start with Web App** (`streamlit run app.py`)
   - Upload sample images
   - Understand predictions
   - Explore interface

2. **Try Single Image Prediction**
   ```bash
   python predict.py --mode image --image test.jpg
   ```

3. **Review Training Code** (`train.py`)
   - Understand CNN architecture
   - Learn about data augmentation
   - Study training process

### For Intermediate Users

1. **Modify Model Architecture**
   - Edit `train.py` → `build_model()` function
   - Try different CNN architectures
   - Experiment with hyperparameters

2. **Add New Categories**
   - Collect images for new waste types
   - Update class labels
   - Retrain model

3. **Deploy to Mobile**
   - Use TFLite model
   - Build Android/iOS app
   - Test on edge devices

### For Advanced Users

1. **Implement Object Detection**
   - Use YOLO or Faster R-CNN
   - Detect multiple waste items
   - Draw bounding boxes

2. **Build API Endpoint**
   - Create Flask/FastAPI server
   - Deploy to cloud (AWS/GCP)
   - Scale with Docker/Kubernetes

3. **IoT Integration**
   - Connect to Raspberry Pi
   - Build smart bin prototype
   - Add sensors and actuators

---

## 📝 Next Steps

After completing the quick start:

✅ **Read Full Documentation**: See `README.md` for details  
✅ **Explore Code**: Review `train.py` and `predict.py`  
✅ **Customize Model**: Modify architecture and parameters  
✅ **Deploy Application**: Try cloud or edge deployment  
✅ **Contribute**: Share improvements via Pull Requests  

---

## 💬 Need Help?

- **Documentation**: Read full `README.md`
- **Issues**: Check GitHub Issues page
- **Contact**: your.email@example.com

---

## 🌟 Pro Tips

1. **Use GPU**: Training is 5-10x faster with GPU
2. **Data Augmentation**: Improves model generalization
3. **Early Stopping**: Prevents overfitting automatically
4. **Transfer Learning**: Faster convergence than training from scratch
5. **TFLite Model**: Use for mobile/edge deployment

---

**Ready to make a difference? Start classifying! ♻️🌍**
