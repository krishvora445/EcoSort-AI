"""
Setup script for EcoSort-AI project
Run this after cloning the repository
"""
import os
import sys
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def check_python_version():
    """Verify Python version is 3.12"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor == 12:
        print("✓ Python version is correct (3.12)")
        return True
    else:
        print(f"⚠ Warning: Expected Python 3.12, got {version.major}.{version.minor}")
        return False

def create_folder_structure():
    """Create necessary project folders"""
    print_header("Creating Folder Structure")
    
    folders = [
        "data/raw/train",
        "data/raw/test",
        "data/processed",
        "models",
        "outputs/plots",
        "outputs/logs"
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {folder}")
    
    # Create .gitkeep files
    gitkeep_folders = ["models", "outputs", "data/processed"]
    for folder in gitkeep_folders:
        gitkeep_path = Path(folder) / ".gitkeep"
        gitkeep_path.touch()

def check_dataset():
    """Check if dataset is present"""
    print_header("Checking Dataset")
    
    train_dir = Path("data/raw/train")
    test_dir = Path("data/raw/test")
    
    if not train_dir.exists() or not test_dir.exists():
        print("⚠ Dataset folders not found!")
        print("\nPlease copy your dataset to:")
        print(f"  - {train_dir.absolute()}")
        print(f"  - {test_dir.absolute()}")
        print("\nExpected structure:")
        print("  data/raw/train/O/  (Organic waste images)")
        print("  data/raw/train/R/  (Recyclable waste images)")
        print("  data/raw/test/O/")
        print("  data/raw/test/R/")
        return False
    
    # Check for images
    o_train = list(train_dir.glob("O/*.jpg")) + list(train_dir.glob("O/*.png"))
    r_train = list(train_dir.glob("R/*.jpg")) + list(train_dir.glob("R/*.png"))
    
    if len(o_train) > 0 and len(r_train) > 0:
        print(f"✓ Found {len(o_train)} Organic training images")
        print(f"✓ Found {len(r_train)} Recyclable training images")
        return True
    else:
        print("⚠ No images found in dataset folders")
        return False

def install_requirements():
    """Install Python dependencies"""
    print_header("Installing Requirements")
    
    if Path("requirements.txt").exists():
        print("Installing packages from requirements.txt...")
        print("Run: pip install -r requirements.txt")
        return True
    else:
        print("⚠ requirements.txt not found!")
        return False

def verify_tensorflow():
    """Verify TensorFlow installation"""
    print_header("Verifying TensorFlow")
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow version: {tf.__version__}")
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU available: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"  - {gpu.name}")
        else:
            print("ℹ No GPU detected - will use CPU")
        
        return True
    except ImportError:
        print("✗ TensorFlow not installed!")
        print("Install with: pip install tensorflow")
        return False

def main():
    """Main setup function"""
    print_header("EcoSort-AI Project Setup")
    
    results = {
        "Python Version": check_python_version(),
        "Folder Structure": True,
        "Dataset": False,
        "Dependencies": False,
        "TensorFlow": False
    }
    
    # Create folders
    create_folder_structure()
    
    # Check dataset
    results["Dataset"] = check_dataset()
    
    # Check requirements
    results["Dependencies"] = install_requirements()
    
    # Try to verify TensorFlow if installed
    try:
        results["TensorFlow"] = verify_tensorflow()
    except:
        pass
    
    # Summary
    print_header("Setup Summary")
    
    for check, status in results.items():
        icon = "✓" if status else "✗"
        print(f"{icon} {check}")
    
    print("\n" + "="*60)
    
    if all(results.values()):
        print("\n✓ Setup complete! Ready to train your model.")
        print("\nNext steps:")
        print("  1. Verify config: python config.py")
        print("  2. Train model: python Core\\ Scripts/train.py")
    else:
        print("\n⚠ Setup incomplete. Please:")
        if not results["Dataset"]:
            print("  - Copy your dataset to data/raw/")
        if not results["Dependencies"]:
            print("  - Run: pip install -r requirements.txt")
        if not results["TensorFlow"]:
            print("  - Install TensorFlow: pip install tensorflow")

if __name__ == "__main__":
    main()
