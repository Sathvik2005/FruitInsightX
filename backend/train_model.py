"""
Comprehensive Fruit Classification Model Training Pipeline
Supports multiple architectures: CNN, ResNet, VGG19, MobileNetV2, InceptionV3
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    ResNet50, VGG19, MobileNetV2, InceptionV3
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Configuration
CONFIG = {
    'image_size': (100, 100),
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'test_split': 0.1,
    'early_stopping_patience': 15,
    'reduce_lr_patience': 7,
    'classes': [
        'Apple', 'Banana', 'Cherry', 'Grape', 'Guava',
        'Kiwi', 'Mango', 'Orange', 'Peach', 'Pear', 'Strawberry'
    ],
    'seed': 42
}

# Set seeds for reproducibility
np.random.seed(CONFIG['seed'])
tf.random.set_seed(CONFIG['seed'])


class FruitModelTrainer:
    """Comprehensive model trainer for fruit classification"""
    
    def __init__(self, data_dir, output_dir='models', architecture='custom_cnn'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.architecture = architecture
        self.model = None
        self.history = None
        self.class_names = CONFIG['classes']
        
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def prepare_data(self):
        """Prepare data generators with augmentation"""
        print("📊 Preparing data generators...")
        
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2],
            validation_split=CONFIG['validation_split']
        )
        
        # Validation/Test data (only rescaling)
        test_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=CONFIG['test_split']
        )
        
        # Training generator
        self.train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=CONFIG['image_size'],
            batch_size=CONFIG['batch_size'],
            class_mode='categorical',
            subset='training',
            seed=CONFIG['seed']
        )
        
        # Validation generator
        self.val_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=CONFIG['image_size'],
            batch_size=CONFIG['batch_size'],
            class_mode='categorical',
            subset='validation',
            seed=CONFIG['seed']
        )
        
        # Save class indices
        self.class_indices = self.train_generator.class_indices
        self.save_class_mapping()
        
        print(f"✓ Found {self.train_generator.samples} training samples")
        print(f"✓ Found {self.val_generator.samples} validation samples")
        print(f"✓ Classes: {list(self.class_indices.keys())}")
        
    def save_class_mapping(self):
        """Save class name to index mapping"""
        mapping_path = os.path.join(self.output_dir, 'class_mapping.json')
        with open(mapping_path, 'w') as f:
            json.dump(self.class_indices, f, indent=2)
        print(f"✓ Class mapping saved to {mapping_path}")
        
    def build_custom_cnn(self):
        """Build custom CNN architecture"""
        print("🏗️ Building Custom CNN...")
        
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(*CONFIG['image_size'], 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        return model
    
    def build_resnet50(self):
        """Build ResNet50 transfer learning model"""
        print("🏗️ Building ResNet50...")
        
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(*CONFIG['image_size'], 3)
        )
        
        # Freeze base layers
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        return model
    
    def build_vgg19(self):
        """Build VGG19 transfer learning model"""
        print("🏗️ Building VGG19...")
        
        base_model = VGG19(
            include_top=False,
            weights='imagenet',
            input_shape=(*CONFIG['image_size'], 3)
        )
        
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        return model
    
    def build_mobilenetv2(self):
        """Build MobileNetV2 for edge deployment"""
        print("🏗️ Building MobileNetV2...")
        
        base_model = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=(*CONFIG['image_size'], 3)
        )
        
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        return model
    
    def build_inceptionv3(self):
        """Build InceptionV3 transfer learning model"""
        print("🏗️ Building InceptionV3...")
        
        base_model = InceptionV3(
            include_top=False,
            weights='imagenet',
            input_shape=(*CONFIG['image_size'], 3)
        )
        
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        return model
    
    def build_model(self):
        """Build model based on architecture selection"""
        architectures = {
            'custom_cnn': self.build_custom_cnn,
            'resnet50': self.build_resnet50,
            'vgg19': self.build_vgg19,
            'mobilenetv2': self.build_mobilenetv2,
            'inceptionv3': self.build_inceptionv3
        }
        
        if self.architecture not in architectures:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        self.model = architectures[self.architecture]()
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        print(f"✓ Model built: {self.architecture}")
        self.model.summary()
        
    def get_callbacks(self):
        """Setup training callbacks"""
        model_name = f"{self.architecture}_{self.timestamp}"
        
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(self.output_dir, f'{model_name}_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=CONFIG['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=CONFIG['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            ),
            TensorBoard(
                log_dir=os.path.join(self.output_dir, 'logs', model_name),
                histogram_freq=1
            ),
            CSVLogger(
                os.path.join(self.output_dir, f'{model_name}_training.csv')
            )
        ]
        
        return callbacks
    
    def train(self):
        """Train the model"""
        print(f"\n🚀 Starting training: {self.architecture}")
        print(f"Epochs: {CONFIG['epochs']}, Batch Size: {CONFIG['batch_size']}")
        
        callbacks = self.get_callbacks()
        
        self.history = self.model.fit(
            self.train_generator,
            epochs=CONFIG['epochs'],
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✓ Training completed!")
        
    def save_model(self):
        """Save final model"""
        model_name = f"{self.architecture}_{self.timestamp}_final.h5"
        model_path = os.path.join(self.output_dir, model_name)
        self.model.save(model_path)
        print(f"✓ Model saved to {model_path}")
        
        # Save as TensorFlow Lite for edge deployment
        if self.architecture in ['mobilenetv2', 'custom_cnn']:
            self.convert_to_tflite()
        
    def convert_to_tflite(self):
        """Convert model to TensorFlow Lite"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        tflite_path = os.path.join(
            self.output_dir, 
            f"{self.architecture}_{self.timestamp}.tflite"
        )
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"✓ TFLite model saved to {tflite_path}")
        
    def plot_training_history(self):
        """Plot training metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss plot
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(
            self.output_dir, 
            f"{self.architecture}_{self.timestamp}_training.png"
        )
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training plots saved to {plot_path}")
        plt.close()
        
    def evaluate(self):
        """Evaluate model on validation set"""
        print("\n📊 Evaluating model...")
        
        # Get predictions
        val_steps = self.val_generator.samples // self.val_generator.batch_size + 1
        predictions = self.model.predict(self.val_generator, steps=val_steps)
        y_pred = np.argmax(predictions, axis=1)
        
        # Get true labels
        y_true = self.val_generator.classes[:len(y_pred)]
        
        # Classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=list(self.class_indices.keys()),
            output_dict=True
        )
        
        # Save report
        report_df = pd.DataFrame(report).transpose()
        report_path = os.path.join(
            self.output_dir, 
            f"{self.architecture}_{self.timestamp}_report.csv"
        )
        report_df.to_csv(report_path)
        print(f"✓ Classification report saved to {report_path}")
        
        # Print summary
        print("\n📈 Performance Metrics:")
        print(f"Accuracy: {report['accuracy']:.4f}")
        print(f"Macro Avg F1: {report['macro avg']['f1-score']:.4f}")
        print(f"Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}")
        
        # Confusion matrix
        self.plot_confusion_matrix(y_true, y_pred)
        
        return report
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=list(self.class_indices.keys()),
            yticklabels=list(self.class_indices.keys())
        )
        plt.title(f'Confusion Matrix - {self.architecture}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        
        cm_path = os.path.join(
            self.output_dir, 
            f"{self.architecture}_{self.timestamp}_confusion_matrix.png"
        )
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {cm_path}")
        plt.close()


def train_all_architectures(data_dir, output_dir='models'):
    """Train all supported architectures"""
    architectures = [
        'custom_cnn',
        'mobilenetv2',  # Best for edge
        'resnet50',
        'vgg19',
        'inceptionv3'
    ]
    
    results = {}
    
    for arch in architectures:
        print(f"\n{'='*60}")
        print(f"Training: {arch.upper()}")
        print(f"{'='*60}\n")
        
        try:
            trainer = FruitModelTrainer(data_dir, output_dir, arch)
            trainer.prepare_data()
            trainer.build_model()
            trainer.train()
            trainer.save_model()
            trainer.plot_training_history()
            report = trainer.evaluate()
            
            results[arch] = {
                'accuracy': report['accuracy'],
                'f1_score': report['weighted avg']['f1-score']
            }
            
        except Exception as e:
            print(f"❌ Error training {arch}: {e}")
            results[arch] = {'error': str(e)}
    
    # Save comparison results
    comparison_path = os.path.join(output_dir, 'model_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for arch, metrics in results.items():
        if 'error' in metrics:
            print(f"{arch}: ERROR - {metrics['error']}")
        else:
            print(f"{arch}: Accuracy={metrics['accuracy']:.4f}, "
                  f"F1={metrics['f1_score']:.4f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Fruit Classifier Models')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to training data directory')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Output directory for models')
    parser.add_argument('--architecture', type=str, default='all',
                       choices=['all', 'custom_cnn', 'resnet50', 'vgg19', 
                               'mobilenetv2', 'inceptionv3'],
                       help='Model architecture to train')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    
    args = parser.parse_args()
    
    # Update config
    CONFIG['epochs'] = args.epochs
    CONFIG['batch_size'] = args.batch_size
    
    print(f"\n🍎 Fruit Classifier Training Pipeline")
    print(f"Data Directory: {args.data_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Architecture: {args.architecture}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}\n")
    
    if args.architecture == 'all':
        train_all_architectures(args.data_dir, args.output_dir)
    else:
        trainer = FruitModelTrainer(
            args.data_dir, 
            args.output_dir, 
            args.architecture
        )
        trainer.prepare_data()
        trainer.build_model()
        trainer.train()
        trainer.save_model()
        trainer.plot_training_history()
        trainer.evaluate()
    
    print("\n✅ Training completed successfully!")
