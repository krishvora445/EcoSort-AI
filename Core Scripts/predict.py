"""
Real-Time Waste Classification - Prediction Script
Based on Chapter 10: Computer Vision Applications

Features:
- Load trained model
- Classify single images
- Real-time webcam classification
- Batch prediction on test images
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import argparse
from datetime import datetime


class WastePredictor:
    """Real-time waste classification predictor"""
    
    def __init__(self, model_path='models/waste_classifier_final.h5', img_size=(224, 224)):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to saved Keras model
            img_size: Input image dimensions
        """
        self.img_size = img_size
        self.class_names = ['Organic', 'Recyclable']
        self.colors = {
            'Organic': (0, 255, 0),      # Green
            'Recyclable': (0, 165, 255)   # Orange
        }
        
        # Load model
        print(f"Loading model from: {model_path}")
        try:
            self.model = keras.models.load_model(model_path)
            print("✓ Model loaded successfully!")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
        
    def preprocess_image(self, image):
        """
        Preprocess image for model input
        Following Chapter 10: Image Processing techniques
        """
        # Resize to model input size
        img = cv2.resize(image, self.img_size)
        
        # Convert BGR to RGB (OpenCV uses BGR by default)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to [0, 1]
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict_image(self, image):
        """
        Predict waste class for a single image
        
        Returns:
            class_name: Predicted class ('Organic' or 'Recyclable')
            confidence: Prediction confidence (0-1)
        """
        try:
            # Preprocess
            processed_img = self.preprocess_image(image)
            
            # Predict
            prediction = self.model.predict(processed_img, verbose=0)[0][0]
            
            # Interpret prediction (sigmoid output)
            if prediction > 0.5:
                class_name = self.class_names[1]  # Recyclable
                confidence = prediction
            else:
                class_name = self.class_names[0]  # Organic
                confidence = 1 - prediction
            
            return class_name, confidence
        
        except Exception as e:
            print(f"Error during prediction: {e}")
            return "Unknown", 0.0
    
    def predict_from_file(self, image_path):
        """Predict from image file"""
        print(f"\nClassifying: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None, None
        
        # Predict
        class_name, confidence = self.predict_image(image)
        
        # Display results
        print("=" * 50)
        print(f"Prediction: {class_name}")
        print(f"Confidence: {confidence*100:.2f}%")
        print("=" * 50)
        
        # Visualize
        self._display_prediction(image, class_name, confidence)
        
        return class_name, confidence
    
    def predict_from_webcam(self):
        """
        Real-time classification using webcam
        Chapter 11: Real-world Sustainability Applications
        """
        print("\n" + "=" * 70)
        print("REAL-TIME WASTE CLASSIFICATION - WEBCAM MODE")
        print("=" * 70)
        print("Instructions:")
        print("  - Position waste item in front of camera")
        print("  - Press 'q' or 'ESC' to quit")
        print("  - Press 's' to save screenshot")
        print("=" * 70)
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            print("Please check:")
            print("  1. Webcam is connected")
            print("  2. No other application is using the webcam")
            print("  3. Webcam permissions are granted")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n✓ Webcam opened successfully!")
        print("Starting classification...\n")
        
        # Initialize variables
        frame_count = 0
        class_name = "Processing..."
        confidence = 0.0
        
        try:
            while True:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                # Predict every 5 frames (for performance)
                if frame_count % 5 == 0:
                    class_name, confidence = self.predict_image(frame)
                
                # Draw results on frame
                color = self.colors.get(class_name, (255, 255, 255))
                
                # Background rectangle for text
                cv2.rectangle(frame, (10, 10), (450, 120), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, 10), (450, 120), color, 2)
                
                # Class name
                cv2.putText(
                    frame, 
                    f"Class: {class_name}", 
                    (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.0, 
                    color, 
                    2
                )
                
                # Confidence
                cv2.putText(
                    frame, 
                    f"Confidence: {confidence*100:.1f}%", 
                    (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 255, 255), 
                    2
                )
                
                # Display frame
                cv2.imshow('Waste Classification - Press Q or ESC to quit', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("\nExiting...")
                    break
                elif key == ord('s'):
                    # Create results directory if it doesn't exist
                    results_dir = Path('results')
                    results_dir.mkdir(exist_ok=True)
                    
                    # Save screenshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = results_dir / f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(str(filename), frame)
                    print(f"✓ Screenshot saved: {filename}")
                
                frame_count += 1
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("✓ Webcam closed")
    
    def predict_batch(self, image_dir):
        """
        Batch prediction on directory of images
        """
        image_dir = Path(image_dir)
        
        if not image_dir.exists():
            print(f"Error: Directory not found: {image_dir}")
            return []
        
        # Find all image files
        image_files = (
            list(image_dir.glob('*.jpg')) + 
            list(image_dir.glob('*.jpeg')) + 
            list(image_dir.glob('*.png'))
        )
        
        if not image_files:
            print(f"No images found in {image_dir}")
            return []
        
        print(f"\nProcessing {len(image_files)} images from {image_dir}")
        print("=" * 70)
        
        results = []
        
        for img_path in image_files:
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"⚠ Could not load: {img_path.name}")
                continue
            
            class_name, confidence = self.predict_image(image)
            
            results.append({
                'filename': img_path.name,
                'prediction': class_name,
                'confidence': confidence
            })
            
            print(f"✓ {img_path.name:30s} -> {class_name:12s} ({confidence*100:.1f}%)")
        
        print("=" * 70)
        print(f"Processed {len(results)} images successfully")
        
        return results
    
    def _display_prediction(self, image, class_name, confidence):
        """Display prediction with visualization"""
        # Resize for display
        display_img = cv2.resize(image, (600, 600))
        
        # Add prediction overlay
        color = self.colors.get(class_name, (255, 255, 255))
        
        # Background rectangle
        cv2.rectangle(display_img, (10, 10), (590, 120), (0, 0, 0), -1)
        cv2.rectangle(display_img, (10, 10), (590, 120), color, 3)
        
        # Prediction text
        cv2.putText(
            display_img, 
            f"Prediction: {class_name}", 
            (20, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.2, 
            color, 
            3
        )
        
        # Confidence text
        cv2.putText(
            display_img, 
            f"Confidence: {confidence*100:.2f}%", 
            (20, 95), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, 
            (255, 255, 255), 
            2
        )
        
        # Display
        cv2.imshow('Waste Classification Result - Press any key to close', display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Main prediction interface"""
    parser = argparse.ArgumentParser(
        description='AI Waste Classification - Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Webcam mode (default)
  python predict.py
  python predict.py --mode webcam

  # Single image classification
  python predict.py --mode image --image path/to/image.jpg

  # Batch processing
  python predict.py --mode batch --dir path/to/images/

  # Use custom model
  python predict.py --model path/to/model.h5
        """
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        default='webcam',
        choices=['webcam', 'image', 'batch'],
        help='Prediction mode (default: webcam)'
    )
    parser.add_argument(
        '--image', 
        type=str, 
        help='Path to single image (required for image mode)'
    )
    parser.add_argument(
        '--dir', 
        type=str, 
        help='Path to directory for batch prediction (required for batch mode)'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='models/waste_classifier_final.h5',
        help='Path to trained model (default: models/waste_classifier_final.h5)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'image' and not args.image:
        parser.error("--image is required when using 'image' mode")
    
    if args.mode == 'batch' and not args.dir:
        parser.error("--dir is required when using 'batch' mode")
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("\nPlease train the model first:")
        print("  python train.py")
        return
    
    # Initialize predictor
    try:
        predictor = WastePredictor(model_path=args.model)
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")
        return
    
    # Run prediction based on mode
    print(f"\nMode: {args.mode.upper()}")
    
    if args.mode == 'webcam':
        predictor.predict_from_webcam()
    
    elif args.mode == 'image':
        predictor.predict_from_file(args.image)
    
    elif args.mode == 'batch':
        results = predictor.predict_batch(args.dir)
        
        # Optionally save results to CSV
        if results:
            import pandas as pd
            df = pd.DataFrame(results)
            csv_path = Path('results') / 'batch_predictions.csv'
            csv_path.parent.mkdir(exist_ok=True)
            df.to_csv(csv_path, index=False)
            print(f"\n✓ Results saved to: {csv_path}")


if __name__ == "__main__":
    main()