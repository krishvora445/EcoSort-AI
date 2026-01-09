"""
AI Waste Classification System - Main Training Script
Based on Workshop: Green Skilling, Sustainability and AI Contributions
Chapter 9: Core Deep Learning Techniques - Convolutional Neural Networks (CNN)

Project: Automated Waste Sorting for Recycling
Problem: Manual waste sorting is inefficient and leads to recyclable materials in landfills
Solution: AI-powered image classification system (Organic vs Recyclable)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import json
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class WasteClassificationModel:
    """
    CNN-based Waste Classification System
    Implements concepts from Chapter 9: CNNs for Green Applications
    """
    
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        """
        Initialize the waste classification model
        
        Args:
            data_dir: Path to dataset directory
            img_size: Input image dimensions (default: 224x224 for MobileNetV2)
            batch_size: Training batch size
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = ['Organic', 'Recyclable']  # Binary classification
        
        # Create results directory
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        
        print("=" * 70)
        print("AI WASTE CLASSIFICATION SYSTEM")
        print("Workshop: Green Skilling & AI for Sustainability")
        print("=" * 70)
        print(f"Image Size: {img_size}")
        print(f"Batch Size: {batch_size}")
        print(f"Classes: {self.class_names}")
        print("=" * 70)
    
    def prepare_data(self):
        """
        Data Preparation using Image Augmentation
        Based on Chapter 10: Computer Vision - Image Processing
        """
        print("\n[STEP 1] DATA PREPARATION")
        print("-" * 70)
        
        # Training data augmentation (Chapter 9: Data Augmentation)
        train_datagen = ImageDataGenerator(
            rescale=1./255,              # Normalize pixel values
            rotation_range=20,           # Random rotation
            width_shift_range=0.2,       # Horizontal shift
            height_shift_range=0.2,      # Vertical shift
            shear_range=0.2,             # Shear transformation
            zoom_range=0.2,              # Random zoom
            horizontal_flip=True,        # Horizontal flip
            fill_mode='nearest',         # Fill strategy
            validation_split=0.2         # 80-20 train-validation split
        )
        
        # Validation/Test data (only rescaling, no augmentation)
        test_datagen = ImageDataGenerator(
            rescale=1./255
        )
        
        # Load training data
        print("\nLoading Training Data...")
        self.train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',         # Binary classification
            subset='training',
            shuffle=True,
            seed=42
        )
        
        # Load validation data
        print("Loading Validation Data...")
        self.val_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation',
            shuffle=False,
            seed=42
        )
        
        print(f"\n✓ Training samples: {self.train_generator.samples}")
        print(f"✓ Validation samples: {self.val_generator.samples}")
        print(f"✓ Classes found: {list(self.train_generator.class_indices.keys())}")
        print("-" * 70)
    
    def build_model(self, use_transfer_learning=True):
        """
        Build CNN Model Architecture
        Based on Chapter 9: Convolutional Neural Networks
        
        Args:
            use_transfer_learning: Use pre-trained MobileNetV2 (recommended)
        """
        print("\n[STEP 2] MODEL BUILDING")
        print("-" * 70)
        
        if use_transfer_learning:
            print("Using Transfer Learning with MobileNetV2")
            print("(Chapter 8: Deep Learning Frameworks - TensorFlow/Keras)")
            
            # Load pre-trained MobileNetV2 (trained on ImageNet)
            base_model = MobileNetV2(
                input_shape=(*self.img_size, 3),
                include_top=False,          # Remove classification head
                weights='imagenet'          # Use ImageNet weights
            )
            
            # Freeze base model layers (transfer learning strategy)
            base_model.trainable = False
            
            # Build custom classification head
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.BatchNormalization(),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.5),         # Regularization
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(1, activation='sigmoid')  # Binary classification
            ], name='Waste_Classifier_MobileNetV2')
            
        else:
            print("Building Custom CNN from Scratch")
            print("(Chapter 9: CNN Architecture Components)")
            
            model = models.Sequential([
                # Convolutional Block 1
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Convolutional Block 2
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Convolutional Block 3
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Convolutional Block 4
                layers.Conv2D(256, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Fully Connected Layers
                layers.Flatten(),
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(1, activation='sigmoid')
            ], name='Waste_Classifier_Custom_CNN')
        
        self.model = model
        
        # Display model architecture
        print("\n" + "=" * 70)
        print("MODEL ARCHITECTURE")
        print("=" * 70)
        self.model.summary()
        print("-" * 70)
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile Model with Loss Function and Optimizer
        Based on Chapter 7: Loss Functions & Optimization
        """
        print("\n[STEP 3] MODEL COMPILATION")
        print("-" * 70)
        
        # Binary Cross-Entropy Loss (for binary classification)
        # Adam Optimizer (adaptive learning rate)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')]
        )
        
        print(f"✓ Optimizer: Adam (lr={learning_rate})")
        print(f"✓ Loss Function: Binary Cross-Entropy")
        print(f"✓ Metrics: Accuracy, Precision, Recall, AUC")
        print("-" * 70)
    
    def train_model(self, epochs=25):
        """
        Train the CNN Model
        Based on Chapter 9: Training Deep Learning Models
        """
        print("\n[STEP 4] MODEL TRAINING")
        print("-" * 70)
        print(f"Training for {epochs} epochs...")
        print("Callbacks: Early Stopping, Model Checkpoint, Learning Rate Reduction")
        print("-" * 70)
        
        # Define callbacks (Chapter 9: Overfitting Prevention Strategies)
        callbacks = [
            # Early Stopping: Stop if validation loss doesn't improve
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model Checkpoint: Save best model
            ModelCheckpoint(
                'models/best_waste_classifier.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce Learning Rate: Reduce LR when plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✓ Training completed!")
        print("-" * 70)
        
        return self.history
    
    def evaluate_model(self):
        """
        Evaluate Model Performance
        Based on Chapter 9: Model Evaluation Metrics
        """
        print("\n[STEP 5] MODEL EVALUATION")
        print("-" * 70)
        
        # Evaluate on validation set
        val_loss, val_acc, val_precision, val_recall, val_auc = self.model.evaluate(
            self.val_generator,
            verbose=1
        )
        
        print("\n" + "=" * 70)
        print("VALIDATION RESULTS")
        print("=" * 70)
        print(f"Loss:      {val_loss:.4f}")
        print(f"Accuracy:  {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"Precision: {val_precision:.4f}")
        print(f"Recall:    {val_recall:.4f}")
        print(f"AUC:       {val_auc:.4f}")
        
        # Calculate F1-Score
        f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall)
        print(f"F1-Score:  {f1_score:.4f}")
        print("=" * 70)
        
        # Generate predictions for confusion matrix
        self.val_generator.reset()
        y_pred_probs = self.model.predict(self.val_generator, verbose=1)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        y_true = self.val_generator.classes
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification Report
        print("\n" + "=" * 70)
        print("CLASSIFICATION REPORT")
        print("=" * 70)
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=self.class_names,
            digits=4
        )
        print(report)
        print("=" * 70)
        
        # Save results
        results = {
            'val_loss': float(val_loss),
            'val_accuracy': float(val_acc),
            'val_precision': float(val_precision),
            'val_recall': float(val_recall),
            'val_auc': float(val_auc),
            'f1_score': float(f1_score),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.results_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        return cm, results
    
    def plot_training_history(self):
        """
        Visualize Training Performance
        Based on Python for Sustainability: Visualization
        """
        print("\n[STEP 6] VISUALIZING TRAINING HISTORY")
        print("-" * 70)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History - Waste Classification CNN', fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision', linewidth=2)
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision', linewidth=2)
        axes[1, 0].set_title('Model Precision', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall', linewidth=2)
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall', linewidth=2)
        axes[1, 1].set_title('Model Recall', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        print(f"✓ Training history saved: {self.results_dir / 'training_history.png'}")
        plt.close()
    
    def plot_confusion_matrix(self, cm):
        """
        Plot Confusion Matrix
        Chapter 9: Model Evaluation Visualization
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Waste Classification', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved: {self.results_dir / 'confusion_matrix.png'}")
        plt.close()
    
    def save_model(self, filename='waste_classifier_final.h5'):
        """Save trained model"""
        model_path = Path('models') / filename
        model_path.parent.mkdir(exist_ok=True)
        self.model.save(model_path)
        print(f"\n✓ Model saved: {model_path}")
        
        # Also save in TensorFlow Lite format for deployment
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        
        tflite_path = Path('models') / filename.replace('.h5', '.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"✓ TFLite model saved: {tflite_path} (for mobile deployment)")


def main():
    """
    Main Training Pipeline
    Following Workshop Lab Structure
    """
    print("\n" + "=" * 70)
    print("AI WASTE CLASSIFICATION - TRAINING PIPELINE")
    print("Based on: Green Skilling & AI for Sustainability Workshop")
    print("=" * 70)
    
    # Configuration
    DATA_DIR = 'data/DATASET'  # Update with your dataset path
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 25
    LEARNING_RATE = 0.001
    
    # Initialize model
    classifier = WasteClassificationModel(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Step 1: Prepare data
    classifier.prepare_data()
    
    # Step 2: Build model (with transfer learning)
    classifier.build_model(use_transfer_learning=True)
    
    # Step 3: Compile model
    classifier.compile_model(learning_rate=LEARNING_RATE)
    
    # Step 4: Train model
    history = classifier.train_model(epochs=EPOCHS)
    
    # Step 5: Evaluate model
    cm, results = classifier.evaluate_model()
    
    # Step 6: Visualize results
    classifier.plot_training_history()
    classifier.plot_confusion_matrix(cm)
    
    # Step 7: Save model
    classifier.save_model()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nNext Steps:")
    print("1. Check 'results/' folder for visualizations")
    print("2. Check 'models/' folder for saved models")
    print("3. Run 'predict.py' for real-time waste classification")
    print("4. Deploy using 'app.py' for web interface")
    print("=" * 70)


if __name__ == "__main__":
    main()
