# """
# Real-Time Waste Classification - Prediction Script
# Based on Chapter 10: Computer Vision Applications

# Features:
# - Load trained model
# - Classify single images
# - Real-time webcam classification
# - Batch prediction on test images
# """

# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from pathlib import Path
# import argparse
# from datetime import datetime


# class WastePredictor:
#     """Real-time waste classification predictor"""
    
#     def __init__(self, model_path='models/waste_classifier_final.h5', img_size=(224, 224)):
#         """
#         Initialize predictor with trained model
        
#         Args:
#             model_path: Path to saved Keras model
#             img_size: Input image dimensions
#         """
#         self.img_size = img_size
#         self.class_names = ['Organic', 'Recyclable']
#         self.colors = {
#             'Organic': (0, 255, 0),      # Green
#             'Recyclable': (0, 165, 255)   # Orange
#         }
        
#         # Load model
#         print(f"Loading model from: {model_path}")
#         self.model = keras.models.load_model(model_path)
#         print("✓ Model loaded successfully!")
        
#     def preprocess_image(self, image):
#         """
#         Preprocess image for model input
#         Following Chapter 10: Image Processing techniques
#         """
#         # Resize to model input size
#         img = cv2.resize(image, self.img_size)
        
#         # Convert BGR to RGB (OpenCV uses BGR by default)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         # Normalize pixel values to [0, 1]
#         img = img.astype('float32') / 255.0
        
#         # Add batch dimension
#         img = np.expand_dims(img, axis=0)
        
#         return img
    
#     def predict_image(self, image):
#         """
#         Predict waste class for a single image
        
#         Returns:
#             class_name: Predicted class ('Organic' or 'Recyclable')
#             confidence: Prediction confidence (0-1)
#         """
#         # Preprocess
#         processed_img = self.preprocess_image(image)
        
#         # Predict
#         prediction = self.model.predict(processed_img, verbose=0)[0][0]
        
#         # Interpret prediction (sigmoid output)
#         if prediction > 0.5:
#             class_name = self.class_names[1]  # Recyclable
#             confidence = prediction
#         else:
#             class_name = self.class_names[0]  # Organic
#             confidence = 1 - prediction
        
#         return class_name, confidence
    
#     def predict_from_file(self, image_path):
#         """Predict from image file"""
#         print(f"\nClassifying: {image_path}")
        
#         # Load image
#         image = cv2.imread(str(image_path))
#         if image is None:
#             print(f"Error: Could not load image from {image_path}")
#             return
        
#         # Predict
#         class_name, confidence = self.predict_image(image)
        
#         # Display results
#         print("=" * 50)
#         print(f"Prediction: {class_name}")
#         print(f"Confidence: {confidence*100:.2f}%")
#         print("=" * 50)
        
#         # Visualize
#         self._display_prediction(image, class_name, confidence)
        
#         return class_name, confidence
    
#     def predict_from_webcam(self):
#         """
#         Real-time classification using webcam
#         Chapter 11: Real-world Sustainability Applications
#         """
#         print("\n" + "=" * 70)
#         print("REAL-TIME WASTE CLASSIFICATION - WEBCAM MODE")
#         print("=" * 70)
#         print("Instructions:")
#         print("  - Position waste item in front of camera")
#         print("  - Press 'q' to quit")
#         print("  - Press 's' to save screenshot")
#         print("=" * 70)
        
#         # Open webcam
#         cap = cv2.VideoCapture(0)
        
#         if not cap.isOpened():
#             print("Error: Could not open webcam")
#             return
        
#         # Set camera resolution
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
#         print("\n✓ Webcam opened successfully!")
#         print("Starting classification...\n")
        
#         frame_count = 0
        
#         while True:
#             # Capture frame
#             ret, frame = cap.read()
#             if not ret:
#                 print("Error: Failed to capture frame")
#                 break
            
#             # Predict every 5 frames (for performance)
#             if frame_count % 5 == 0:
#                 class_name, confidence = self.predict_image(frame)
                
#                 # Draw results on frame
#                 color = self.colors[class_name]
                
#                 # Background rectangle for text
#                 cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 0), -1)
                
#                 # Class name
#                 cv2.putText(frame, f"Class: {class_name}", 
#                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
#                            1.0, color, 2)
                
#                 # Confidence
#                 cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", 
#                            (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 
#                            0.7, (255, 255, 255), 2)
            
#             # Display frame
#             cv2.imshow('Waste Classification - Press Q to quit', frame)
            
#             # Handle key presses
#             key = cv2.waitKey(1) & 0xFF
            
#             if key == ord('q'):
#                 print("\nExiting...")
#                 break
#             elif key == ord('s'):
#                 # Save screenshot
#                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                 filename = f"results/screenshot_{timestamp}.jpg"
#                 cv2.imwrite(filename, frame)
#                 print(f"✓ Screenshot saved: {filename}")
            
#             frame_count += 1
        
#         # Cleanup
#         cap.release()
#         cv2.destroyAllWindows()
#         print("✓ Webcam closed")
    
#     def predict_batch(self, image_dir):
#         """
#         Batch prediction on directory of images
#         """
#         image_dir = Path(image_dir)
#         image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        
#         if not image_files:
#             print(f"No images found in {image_dir}")
#             return
        
#         print(f"\nProcessing {len(image_files)} images from {image_dir}")
#         print("=" * 70)
        
#         results = []
        
#         for img_path in image_files:
#             image = cv2.imread(str(img_path))
#             if image is None:
#                 continue
            
#             class_name, confidence = self.predict_image(image)
            
#             results.append({
#                 'filename': img_path.name,
#                 'prediction': class_name,
#                 'confidence': confidence
#             })
            
#             print(f"✓ {img_path.name:30s} -> {class_name:12s} ({confidence*100:.1f}%)")
        
#         print("=" * 70)
#         print(f"Processed {len(results)} images")
        
#         return results
    
#     def _display_prediction(self, image, class_name, confidence):
#         """Display prediction with visualization"""
#         # Resize for display
#         display_img = cv2.resize(image, (600, 600))
        
#         # Add prediction overlay
#         color = self.colors[class_name]
#         cv2.rectangle(display_img, (10, 10), (590, 100), (0, 0, 0), -1)
#         cv2.putText(display_img, f"Prediction: {class_name}", 
#                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
#         cv2.putText(display_img, f"Confidence: {confidence*100:.2f}%", 
#                    (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
#         # Display
#         cv2.imshow('Waste Classification Result', display_img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()


# def main():
#     """Main prediction interface"""
#     parser = argparse.ArgumentParser(description='Waste Classification Predictor')
#     parser.add_argument('--mode', type=str, default='webcam', 
#                        choices=['webcam', 'image', 'batch'],
#                        help='Prediction mode')
#     parser.add_argument('--image', type=str, help='Path to single image')
#     parser.add_argument('--dir', type=str, help='Path to directory for batch prediction')
#     parser.add_argument('--model', type=str, default='models/waste_classifier_final.h5',
#                        help='Path to trained model')
    
#     args = parser.parse_args()
    
#     # Initialize predictor
#     predictor = WastePredictor(model_path=args.model)
    
#     # Run prediction based on mode
#     if args.mode == 'webcam':
#         predictor.predict_from_webcam()
#     elif args.mode == 'image':
#         if not args.image:
#             print("Error: Please provide --image path")
#             return
#         predictor.predict_from_file(args.image)
#     elif args.mode == 'batch':
#         if not args.dir:
#             print("Error: Please provide --dir path")
#             return
#         predictor.predict_batch(args.dir)


# if __name__ == "__main__":
#     main()




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
        self.model = keras.models.load_model(model_path)
        print("✓ Model loaded successfully!")
        
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
    
    def predict_from_file(self, image_path):
        """Predict from image file"""
        print(f"\nClassifying: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return
        
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
        print("  - Press 'q' to quit")
        print("  - Press 's' to save screenshot")
        print("=" * 70)
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n✓ Webcam opened successfully!")
        print("Starting classification...\n")
        
        frame_count = 0
        
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
                color = self.colors[class_name]
                
                # Background rectangle for text
                cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 0), -1)
                
                # Class name
                cv2.putText(frame, f"Class: {class_name}", 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, color, 2)
                
                # Confidence
                cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", 
                           (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Waste Classification - Press Q to quit', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nExiting...")
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"results/screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"✓ Screenshot saved: {filename}")
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Webcam closed")
    
    def predict_batch(self, image_dir):
        """
        Batch prediction on directory of images
        """
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        
        if not image_files:
            print(f"No images found in {image_dir}")
            return
        
        print(f"\nProcessing {len(image_files)} images from {image_dir}")
        print("=" * 70)
        
        results = []
        
        for img_path in image_files:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            class_name, confidence = self.predict_image(image)
            
            results.append({
                'filename': img_path.name,
                'prediction': class_name,
                'confidence': confidence
            })
            
            print(f"✓ {img_path.name:30s} -> {class_name:12s} ({confidence*100:.1f}%)")
        
        print("=" * 70)
        print(f"Processed {len(results)} images")
        
        return results
    
    def _display_prediction(self, image, class_name, confidence):
        """Display prediction with visualization"""
        # Resize for display
        display_img = cv2.resize(image, (600, 600))
        
        # Add prediction overlay
        color = self.colors[class_name]
        cv2.rectangle(display_img, (10, 10), (590, 100), (0, 0, 0), -1)
        cv2.putText(display_img, f"Prediction: {class_name}", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(display_img, f"Confidence: {confidence*100:.2f}%", 
                   (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Display
        cv2.imshow('Waste Classification Result', display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Main prediction interface"""
    parser = argparse.ArgumentParser(description='Waste Classification Predictor')
    parser.add_argument('--mode', type=str, default=None, 
                       choices=['webcam', 'image', 'batch'],
                       help='Prediction mode (auto-detected if not specified)')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--dir', type=str, help='Path to directory for batch prediction')
    parser.add_argument('--model', type=str, default='models/waste_classifier_final.h5',
                       help='Path to trained model')
    
    args = parser.parse_args()
    
    # Auto-detect mode if not specified
    if args.mode is None:
        if args.image:
            args.mode = 'image'
            print("Auto-detected mode: image")
        elif args.dir:
            args.mode = 'batch'
            print("Auto-detected mode: batch")
        else:
            args.mode = 'webcam'
            print("Auto-detected mode: webcam (default)")
    
    # Initialize predictor
    predictor = WastePredictor(model_path=args.model)
    
    # Run prediction based on mode
    if args.mode == 'webcam':
        predictor.predict_from_webcam()
    elif args.mode == 'image':
        if not args.image:
            print("Error: Please provide --image path")
            return
        predictor.predict_from_file(args.image)
    elif args.mode == 'batch':
        if not args.dir:
            print("Error: Please provide --dir path")
            return
        predictor.predict_batch(args.dir)


if __name__ == "__main__":
    main()