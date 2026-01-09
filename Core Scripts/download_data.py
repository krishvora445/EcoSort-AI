"""
Download Waste Classification Dataset from Kaggle
Automated data preparation script
"""

import os
import zipfile
from pathlib import Path
import shutil


def setup_kaggle():
    """
    Setup Kaggle API credentials
    
    Instructions:
    1. Go to https://www.kaggle.com/account
    2. Scroll to 'API' section
    3. Click 'Create New API Token'
    4. Download kaggle.json
    5. Place it in ~/.kaggle/ directory
    """
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        print("=" * 70)
        print("KAGGLE API SETUP REQUIRED")
        print("=" * 70)
        print("\nTo download the dataset, you need to set up Kaggle API:")
        print("\n1. Go to: https://www.kaggle.com/account")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. Download kaggle.json")
        print(f"5. Move it to: {kaggle_dir}")
        print("\nThen run this script again.")
        print("=" * 70)
        return False
    
    # Set permissions (Unix-based systems)
    try:
        os.chmod(kaggle_json, 0o600)
    except:
        pass
    
    return True


def download_dataset():
    """Download waste classification dataset from Kaggle"""
    
    print("\n" + "=" * 70)
    print("DOWNLOADING WASTE CLASSIFICATION DATASET")
    print("=" * 70)
    
    # Check Kaggle setup
    if not setup_kaggle():
        return False
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Initialize API
        api = KaggleApi()
        api.authenticate()
        
        # Dataset details
        dataset = 'techsash/waste-classification-data'
        
        # Create data directory
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        
        print(f"\n📥 Downloading dataset: {dataset}")
        print("This may take a few minutes (~2-3 GB)...")
        
        # Download dataset
        api.dataset_download_files(
            dataset,
            path=str(data_dir),
            unzip=True
        )
        
        print("\n✓ Dataset downloaded successfully!")
        
        # Verify dataset structure
        dataset_path = data_dir / 'DATASET'
        if dataset_path.exists():
            print("\n📁 Dataset Structure:")
            for class_dir in dataset_path.iterdir():
                if class_dir.is_dir():
                    num_images = len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png')))
                    print(f"  - {class_dir.name}: {num_images} images")
            
            print("\n" + "=" * 70)
            print("DATASET READY FOR TRAINING!")
            print("=" * 70)
            print("\nNext steps:")
            print("1. Review the data in 'data/DATASET/'")
            print("2. Run: python train.py")
            print("=" * 70)
            return True
        else:
            print("\n⚠️ Warning: Dataset structure not as expected")
            print(f"Expected path: {dataset_path}")
            return False
            
    except ImportError:
        print("\n❌ Error: Kaggle API not installed")
        print("Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nAlternative: Download manually from:")
        print("https://www.kaggle.com/datasets/techsash/waste-classification-data")
        return False


def verify_dataset_structure(data_dir='data/DATASET'):
    """Verify and display dataset statistics"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Dataset not found at: {data_path}")
        return False
    
    print("\n" + "=" * 70)
    print("DATASET VERIFICATION")
    print("=" * 70)
    
    total_images = 0
    
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            num_images = len(images)
            total_images += num_images
            
            print(f"\n📁 {class_dir.name}:")
            print(f"   Images: {num_images}")
            
            if num_images > 0:
                # Show first few image names
                print(f"   Sample files:")
                for img in images[:3]:
                    print(f"     - {img.name}")
    
    print(f"\n📊 Total Images: {total_images}")
    print("=" * 70)
    
    return total_images > 0


def create_sample_split(data_dir='data/DATASET', test_split=0.1):
    """
    Optional: Create a separate test set
    (Training script uses 80-20 train-validation split)
    """
    import random
    
    data_path = Path(data_dir)
    test_dir = Path('data/TEST')
    
    if test_dir.exists():
        print("\n⚠️ Test directory already exists")
        return
    
    print("\n" + "=" * 70)
    print(f"CREATING TEST SET ({test_split*100}% of data)")
    print("=" * 70)
    
    test_dir.mkdir(parents=True, exist_ok=True)
    
    for class_dir in data_path.iterdir():
        if not class_dir.is_dir():
            continue
        
        # Get all images
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        
        # Shuffle
        random.seed(42)
        random.shuffle(images)
        
        # Calculate split
        num_test = int(len(images) * test_split)
        test_images = images[:num_test]
        
        # Create test class directory
        test_class_dir = test_dir / class_dir.name
        test_class_dir.mkdir(exist_ok=True)
        
        # Move images
        for img in test_images:
            shutil.copy(img, test_class_dir / img.name)
        
        print(f"✓ {class_dir.name}: {num_test} images moved to test set")
    
    print("\n✓ Test set created!")
    print(f"Location: {test_dir}")
    print("=" * 70)


def main():
    """Main data preparation pipeline"""
    print("\n" + "=" * 70)
    print("WASTE CLASSIFICATION - DATA PREPARATION")
    print("=" * 70)
    
    # Download dataset
    success = download_dataset()
    
    if success:
        # Verify structure
        verify_dataset_structure()
        
        # Optional: Create test split
        create_test = input("\nCreate separate test set? (y/n): ").lower()
        if create_test == 'y':
            create_sample_split()
    
    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
