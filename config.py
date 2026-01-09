"""
Configuration file for EcoSort-AI Waste Classification Project
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset paths
TRAIN_DIR = RAW_DATA_DIR / "train"
TEST_DIR = RAW_DATA_DIR / "test"

# Class names
CLASSES = {
    'O': 'Organic',
    'R': 'Recyclable'
}

# Model directory
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "waste_classifier.h5"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
LOGS_DIR = OUTPUT_DIR / "logs"

# Training parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Image preprocessing
IMG_CHANNELS = 3
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
NORMALIZE_STD = [0.229, 0.224, 0.225]   # ImageNet std

# Data augmentation parameters
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'zoom_range': 0.2,
    'shear_range': 0.15,
    'fill_mode': 'nearest'
}

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODEL_DIR,
        OUTPUT_DIR,
        PLOTS_DIR,
        LOGS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created/verified: {directory}")

def verify_dataset():
    """Verify that dataset exists and has correct structure"""
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Training data not found at {TRAIN_DIR}\n"
            "Please ensure your dataset is in data/raw/train/ folder"
        )
    
    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test data not found at {TEST_DIR}\n"
            "Please ensure your dataset is in data/raw/test/ folder"
        )
    
    # Check for class folders
    for split_dir in [TRAIN_DIR, TEST_DIR]:
        for class_code in CLASSES.keys():
            class_dir = split_dir / class_code
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Class folder not found: {class_dir}\n"
                    f"Expected structure: {split_dir}/{{O,R}}/"
                )
    
    print("✓ Dataset structure verified!")
    
    # Count images
    for split_name, split_dir in [("Training", TRAIN_DIR), ("Testing", TEST_DIR)]:
        print(f"\n{split_name} set:")
        for class_code, class_name in CLASSES.items():
            class_dir = split_dir / class_code
            img_count = len(list(class_dir.glob("*.jpg"))) + \
                       len(list(class_dir.glob("*.png"))) + \
                       len(list(class_dir.glob("*.jpeg")))
            print(f"  {class_name} ({class_code}): {img_count} images")

if __name__ == "__main__":
    print("Setting up EcoSort-AI project structure...")
    create_directories()
    verify_dataset()
    print("\n✓ Project setup complete!")
