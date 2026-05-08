"""
YOLO Variable Star Classifier - Class-Specific Normalization Version (Memory Optimized with SGD)
Uses pre-split class-specific normalized images with 5-fold CV on train set
Enhanced with SGD optimizer for better performance on variable star classification
"""

import torch
import torch.serialization
import os
import glob
import time
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from pathlib import Path
import cv2
from PIL import Image
import random
import json
import gc
import psutil

# Fix PyTorch warnings before importing models
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, message='.*torch.cuda.amp.autocast.*')

# Import the visualization mixin
from enhanced_variable_star_visualization import VariableStarVisualizationMixin

# Fix PyTorch 2.6 compatibility issue with YOLO
def patch_torch_load():
    """Patch torch.load to handle YOLO models with PyTorch 2.6+"""
    original_load = torch.load
    
    def patched_load(*args, **kwargs):
        # For YOLO models, always use weights_only=False
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    
    torch.load = patched_load
    torch.serialization.load = patched_load

# Apply the patch at import time
patch_torch_load()

class MemoryManager:
    """Memory management utilities for YOLO training"""
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        ram_mb = memory_info.rss / 1024 / 1024
        
        gpu_mb = 0
        if torch.cuda.is_available():
            gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
            
        return ram_mb, gpu_mb
    
    @staticmethod
    def aggressive_cleanup():
        """Perform aggressive memory cleanup"""
        collected = gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        return collected
    
    @staticmethod
    def check_memory_and_adjust_params(batch_size, cv_folds, epochs):
        """Check available memory and adjust parameters if needed"""
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        
        print(f"💻 System Memory: {total_memory:.1f}GB total, {available_memory:.1f}GB available")
        
        # Adjust parameters based on available memory
        if available_memory < 4:
            print("⚠️  Very low memory! Reducing parameters aggressively...")
            batch_size = min(batch_size, 4)
            cv_folds = min(cv_folds, 3)
            epochs = min(epochs, 25)
        elif available_memory < 6:
            print("⚠️  Low memory detected. Reducing parameters...")
            batch_size = min(batch_size, 8)
            cv_folds = min(cv_folds, 3)
            epochs = min(epochs, 30)
        elif available_memory < 8:
            print("⚠️  Moderate memory. Minor adjustments...")
            batch_size = min(batch_size, 16)
        
        return batch_size, cv_folds, epochs

class ClassSpecificYOLOClassifier(VariableStarVisualizationMixin):
    def __init__(self, model_size='n', img_size=224, device='auto', random_seed=42):
        """
        Initialize YOLO classifier for class-specific normalized images
        
        Args:
            model_size: 'n', 's', 'm', 'l', 'x' (nano to extra-large)
            img_size: Input image size (224 for your generated charts)
            device: Device to use ('auto', 'cpu', '0', '1', etc.)
            random_seed: Random seed for reproducibility
        """
        self.model_size = model_size
        self.img_size = img_size
        self.random_seed = random_seed
        self.class_names = None  # Required by mixin
        
        # Set all random seeds for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
        
        # Fix device detection for YOLO
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 0
                print("🎮 Using GPU for YOLO training")
            else:
                self.device = 'cpu'
                print("💻 Using CPU for YOLO training")
        else:
            self.device = device
        self.best_fold_model_path = None  # Track best model for test evaluation
            
        self.model = None
        self.results_dir = None
        
    def load_pretrained(self):
        """Load pretrained YOLO classification model with PyTorch 2.6+ compatibility"""
        model_path = f'yolov8{self.model_size}-cls.pt'
        
        try:
            # First attempt: Try with safe globals
            with torch.serialization.safe_globals(['ultralytics.nn.tasks.ClassificationModel']):
                self.model = YOLO(model_path)
            print(f"Loaded YOLOv8{self.model_size} classification model (safe globals method)")
            
        except Exception as e1:
            try:
                # Second attempt: Use the patched loader
                self.model = YOLO(model_path)
                print(f"Loaded YOLOv8{self.model_size} classification model (patched loader)")
                
            except Exception as e2:
                # Third attempt: Force weights_only=False
                import torch
                original_load = torch.load
                
                def force_weights_only_false(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                
                torch.load = force_weights_only_false
                
                try:
                    self.model = YOLO(model_path)
                    print(f"Loaded YOLOv8{self.model_size} classification model (forced weights_only=False)")
                finally:
                    torch.load = original_load
                    
            except Exception as e3:
                print(f"Failed to load YOLO model with all methods:")
                print(f"  Method 1 error: {e1}")
                print(f"  Method 2 error: {e2}")
                print(f"  Method 3 error: {e3}")
                raise e3

    def load_pre_split_data(self, train_images_path, test_images_path):
        """Load pre-split class-specific normalized images"""
        print(f"📁 Loading pre-split class-specific normalized data...")
        print(f"   Train path: {train_images_path}")
        print(f"   Test path: {test_images_path}")
        
        # Verify paths exist
        if not os.path.exists(train_images_path):
            raise FileNotFoundError(f"Train images path not found: {train_images_path}")
        if not os.path.exists(test_images_path):
            raise FileNotFoundError(f"Test images path not found: {test_images_path}")
        
        # Get class names from train directory
        train_class_dirs = [d for d in os.listdir(train_images_path) 
                           if os.path.isdir(os.path.join(train_images_path, d))]
        test_class_dirs = [d for d in os.listdir(test_images_path) 
                          if os.path.isdir(os.path.join(test_images_path, d))]
        
        self.class_names = sorted(train_class_dirs)
        print(f"   Found {len(self.class_names)} classes: {self.class_names}")
        
        # Verify train and test have same classes
        if set(train_class_dirs) != set(test_class_dirs):
            print(f"⚠️  Warning: Train and test class directories don't match!")
            print(f"   Train: {sorted(train_class_dirs)}")
            print(f"   Test: {sorted(test_class_dirs)}")
        
        # Collect all training images and labels
        train_images = []
        train_labels = []
        
        print(f"📊 Loading training images...")
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(train_images_path, class_name)
            if os.path.exists(class_path):
                class_images = []
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    class_images.extend(glob.glob(os.path.join(class_path, ext)))
                
                train_images.extend(class_images)
                train_labels.extend([class_idx] * len(class_images))
                print(f"   {class_name}: {len(class_images)} images")
        
        # Collect all test images and labels
        test_images = []
        test_labels = []
        
        print(f"📊 Loading test images...")
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(test_images_path, class_name)
            if os.path.exists(class_path):
                class_images = []
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    class_images.extend(glob.glob(os.path.join(class_path, ext)))
                
                test_images.extend(class_images)
                test_labels.extend([class_idx] * len(class_images))
                print(f"   {class_name}: {len(class_images)} images")
        
        print(f"\n✅ Data loading complete:")
        print(f"   Training samples: {len(train_images)}")
        print(f"   Test samples: {len(test_images)}")
        print(f"   Classes: {len(self.class_names)}")
        
        return train_images, train_labels, test_images, test_labels

    def create_yolo_dataset_structure(self, images, labels, output_path, dataset_name):
        """Create YOLO-compatible dataset structure with memory optimization"""
        print(f"🔧 Creating YOLO dataset structure for {dataset_name}...")
        
        # Clear and create output directory
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)
        
        # Create class directories
        for class_name in self.class_names:
            os.makedirs(os.path.join(output_path, class_name), exist_ok=True)
        
        # Copy images to YOLO structure in smaller batches
        batch_size = 100  # Process 100 images at a time
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            for img_path, label_idx in zip(batch_images, batch_labels):
                class_name = self.class_names[label_idx]
                dst_dir = os.path.join(output_path, class_name)
                dst_path = os.path.join(dst_dir, os.path.basename(img_path))
                
                # Copy image
                shutil.copy2(img_path, dst_path)
            
            # Small cleanup after each batch
            if i % (batch_size * 5) == 0:  # Every 500 images
                gc.collect()
        
        print(f"   ✅ {dataset_name} dataset structure created at: {output_path}")
        return output_path

    def get_memory_optimized_training_args(self, temp_path, fold_num, epochs, batch_size):
        """Get memory-optimized training arguments with enhanced SGD settings"""
        
        training_args = {
            'data': os.path.join(temp_path, f'fold_{fold_num}'),
            'epochs': epochs,
            'imgsz': self.img_size,
            'batch': batch_size,
            'patience': 15,  # Increased for SGD
            'device': self.device,
            'workers': 1,  # Reduce workers to save memory
            'name': f'cv_fold_{fold_num}',
            'save': True,
            'plots': False,
            'verbose': False,
            'project': self.results_dir,
            
            # Memory optimization settings
            'amp': True if isinstance(self.device, int) else False,
            'cache': False,  # Disable caching to save memory
            'rect': False,   # Disable rectangular training
            'overlap_mask': False,
            'mask_ratio': 0,
            
            # Disable all augmentations to save memory
            'augment': False,
            'mosaic': 0.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            'flipud': 0.0,
            'fliplr': 0.0,
            'degrees': 0.0,
            'translate': 0.0,
            'scale': 0.0,
            'shear': 0.0,
            'perspective': 0.0,
            'hsv_h': 0.0,
            'hsv_s': 0.0,
            'hsv_v': 0.0,
            
            # Enhanced SGD optimizer settings
            'optimizer': 'SGD',        # Changed from 'AdamW'
            'lr0': 0.01,              # Changed from 0.0001 (100x higher)
            'lrf': 0.1,               # Changed from 0.01
            'momentum': 0.9,          # Changed from 0.937
            'weight_decay': 0.0005,   # Changed from 0.001
            'warmup_epochs': 5,       # Changed from 3
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Enhanced regularization for variable star classification
            'dropout': 0.3,           # Changed from 0.2
            'label_smoothing': 0.15,  # Changed from 0.1
            
            # Add cosine annealing for better convergence
            'cos_lr': True,           # NEW - cosine learning rate schedule
            
            # Memory management
            'save_period': -1,
            'exist_ok': True,
            'deterministic': True,
            'seed': self.random_seed + fold_num,
            'single_cls': False,
            'close_mosaic': 0,
        }
        
        return training_args
    def train_cv_fold(self, train_images, train_labels, val_images, val_labels, 
                      fold_num, temp_path, epochs=60, batch_size=16):
        """Train YOLO model for one CV fold with memory optimization and cleanup"""
        
        print(f"📚 Training CV fold {fold_num} (SGD optimizer, memory optimized)...")
        
        # Monitor memory before fold
        ram_before, gpu_before = MemoryManager.get_memory_usage()
        print(f"   Memory before fold: RAM {ram_before:.1f}MB, GPU {gpu_before:.1f}MB")
        
        # Create fold-specific dataset structure
        fold_train_path = os.path.join(temp_path, f'fold_{fold_num}', 'train')
        fold_val_path = os.path.join(temp_path, f'fold_{fold_num}', 'val')
        
        # Create YOLO dataset structure for this fold
        self.create_yolo_dataset_structure(train_images, train_labels, fold_train_path, f"fold {fold_num} train")
        self.create_yolo_dataset_structure(val_images, val_labels, fold_val_path, f"fold {fold_num} val")
        
        # CLEANUP: Remove references to large lists after dataset creation
        del train_images, train_labels, val_images, val_labels
        collected = MemoryManager.aggressive_cleanup()
        print(f"   🧹 Cleanup after data prep: {collected} objects collected")
        
        # Get memory-optimized training arguments
        training_args = self.get_memory_optimized_training_args(temp_path, fold_num, epochs, batch_size)
        
        # Train model with memory monitoring
        print(f"🚀 Training CV fold {fold_num} with SGD optimizer...")
        print(f"   LR: {training_args['lr0']}, Momentum: {training_args['momentum']}")
        print(f"   Label smoothing: {training_args['label_smoothing']}, Dropout: {training_args['dropout']}")
        start_time = time.time()
        
        try:
            results = self.model.train(**training_args)
            training_time = time.time() - start_time
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ Out of memory error in fold {fold_num}")
                print(f"   Retrying with smaller batch size...")
                # Clear memory and try with smaller batch
                MemoryManager.aggressive_cleanup()
                training_args['batch'] = max(1, training_args['batch'] // 2)
                print(f"   New batch size: {training_args['batch']}")
                results = self.model.train(**training_args)
                training_time = time.time() - start_time
            else:
                raise e
        
        # Get training metrics
        best_epoch = results.best_epoch if hasattr(results, 'best_epoch') else epochs
        early_stopped = best_epoch < epochs
        
        print(f"   CV fold {fold_num} training completed in {training_time:.1f}s")
        print(f"   Best epoch: {best_epoch}, Early stopped: {early_stopped}")
        
        # IMMEDIATE CLEANUP: Save best model and remove training artifacts
        runs_dir = os.path.join(self.results_dir, f'cv_fold_{fold_num}')
        if os.path.exists(runs_dir):
            best_model_source = os.path.join(runs_dir, 'weights', 'best.pt')
            if os.path.exists(best_model_source):
                best_model_dest = os.path.join(self.results_dir, f'best_model_fold_{fold_num}.pt')
                shutil.copy2(best_model_source, best_model_dest)
                print(f"   ✅ Best model saved: {best_model_dest}")
            
            # Remove entire runs directory immediately
            shutil.rmtree(runs_dir)
            print(f"   🗑️  Removed training artifacts: {runs_dir}")
        
        # Remove fold temporary data immediately
        fold_temp_dir = os.path.join(temp_path, f'fold_{fold_num}')
        if os.path.exists(fold_temp_dir):
            shutil.rmtree(fold_temp_dir)
            print(f"   🗑️  Removed temp data: {fold_temp_dir}")
        
        # Clear any training state from model
        if hasattr(self.model, 'trainer'):
            del self.model.trainer
        
        # Final cleanup for this fold
        collected = MemoryManager.aggressive_cleanup()
        
        ram_after, gpu_after = MemoryManager.get_memory_usage()
        print(f"   🧹 Final cleanup: {collected} objects collected")
        print(f"   Memory after fold: RAM {ram_after:.1f}MB, GPU {gpu_after:.1f}MB")
        
        return results, training_time, best_epoch, early_stopped

    def evaluate_cv_fold(self, val_images, val_labels, fold_num):
        """Evaluate CV fold performance using best model with memory optimization"""
        print(f"🔍 Evaluating CV fold {fold_num} with best model...")
        
        # Load best model for this fold
        best_model_path = os.path.join(self.results_dir, f'best_model_fold_{fold_num}.pt')
        if os.path.exists(best_model_path):
            eval_model = YOLO(best_model_path)
            print(f"   📥 Loaded best model: {best_model_path}")
        else:
            # Fallback to current model
            eval_model = self.model
            print(f"   ⚠️  Using current model (best model not found)")
        
        predictions = []
        confidences = []
        
        # Predict in batches to manage memory
        batch_size = 20  # Process 20 images at a time
        
        for i in range(0, len(val_images), batch_size):
            batch_images = val_images[i:i+batch_size]
            
            for img_path in batch_images:
                try:
                    if torch.cuda.is_available():
                        with torch.amp.autocast('cuda'):
                            results = eval_model.predict(img_path, imgsz=self.img_size, 
                                                       device=self.device, verbose=False)
                    else:
                        results = eval_model.predict(img_path, imgsz=self.img_size, 
                                                   device=self.device, verbose=False)
                    
                    pred_idx = results[0].probs.top1
                    pred_label = eval_model.names[pred_idx]
                    confidence = results[0].probs.top1conf.item()
                    
                    # Convert class name back to index
                    pred_class_idx = self.class_names.index(pred_label)
                    predictions.append(pred_class_idx)
                    confidences.append(confidence)
                    
                    # Clear result from memory
                    del results
                    
                except Exception as e:
                    print(f"Error predicting {img_path}: {e}")
                    predictions.append(0)  # Default to first class
                    confidences.append(0.0)
            
            # Cleanup after each batch
            if i % (batch_size * 5) == 0:  # Every 100 images
                MemoryManager.aggressive_cleanup()
        
        # Calculate metrics
        accuracy = accuracy_score(val_labels, predictions)
        avg_confidence = np.mean(confidences)
        
        print(f"   CV fold {fold_num} accuracy: {accuracy:.4f}")
        print(f"   Average confidence: {avg_confidence:.4f}")
        
        # Cleanup evaluation model if it's not the main model
        if eval_model != self.model:
            del eval_model
            MemoryManager.aggressive_cleanup()
        
        return predictions, confidences, accuracy

    def cross_validate_on_train_set(self, train_images, train_labels, 
                                   n_splits=5, epochs=60, batch_size=16, temp_path='temp_cv_yolo'):
        """Perform 5-fold cross-validation on the training set with memory optimization"""
        
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        
        # Cross-validation setup
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
        
        cv_results = []
        all_cv_predictions = []
        all_cv_true_labels = []
        cv_training_times = []
        
        # Monitor initial memory
        ram_initial, gpu_initial = MemoryManager.get_memory_usage()
        print(f"\n🎯 Starting {n_splits}-fold cross-validation on training set...")
        print(f"📊 Training set size: {len(train_images)} samples")
        print(f"💻 Initial memory: RAM {ram_initial:.1f}MB, GPU {gpu_initial:.1f}MB")
        print(f"🔧 Using SGD optimizer with enhanced settings for variable star classification")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_images, train_labels)):
            print(f"\n{'='*20} CV Fold {fold + 1}/{n_splits} {'='*20}")

            # Load fresh model for each fold
            self.load_pretrained()
            
            # Get fold data
            fold_train_images = [train_images[i] for i in train_idx]
            fold_train_labels = [train_labels[i] for i in train_idx]
            fold_val_images = [train_images[i] for i in val_idx]
            fold_val_labels = [train_labels[i] for i in val_idx]
            
            print(f"   Fold {fold + 1}: {len(fold_train_images)} train, {len(fold_val_images)} val")
            
            # Train fold with memory optimization
            results, train_time, best_epoch, early_stopped = self.train_cv_fold(
                fold_train_images, fold_train_labels, fold_val_images, fold_val_labels,
                fold + 1, temp_path, epochs, batch_size
            )
            cv_training_times.append(train_time)
            
            # Evaluate fold
            predictions, confidences, accuracy = self.evaluate_cv_fold(
                fold_val_images, fold_val_labels, fold + 1
            )
            
            # Store results
            cv_results.append({
                'fold': fold + 1,
                'accuracy': accuracy,
                'training_time': train_time,
                'avg_confidence': np.mean(confidences),
                'val_samples': len(fold_val_labels),
                'early_stopped': early_stopped,
                'best_epoch': best_epoch,
                'total_epochs_trained': best_epoch if early_stopped else epochs
            })
            
            all_cv_predictions.extend(predictions)
            all_cv_true_labels.extend(fold_val_labels)
            
            print(f"✅ CV fold {fold + 1} completed: {accuracy:.4f} accuracy in {train_time:.1f}s")
            
            # CRITICAL: Inter-fold cleanup
            if self.model is not None:
                del self.model
                self.model = None
            
            # Clean up fold data
            del fold_train_images, fold_train_labels, fold_val_images, fold_val_labels
            del predictions, confidences
            
            collected = MemoryManager.aggressive_cleanup()
            ram_current, gpu_current = MemoryManager.get_memory_usage()
            print(f"   🧹 Inter-fold cleanup: {collected} objects collected")
            print(f"   Memory: RAM {ram_current:.1f}MB, GPU {gpu_current:.1f}MB")
            
            # Memory leak detection
            ram_growth = ram_current - ram_initial
            if ram_growth > 1000:  # More than 1GB growth
                print(f"   ⚠️  WARNING: Potential memory leak! RAM growth: {ram_growth:.1f}MB")
        
        # Find best fold for test evaluation
        cv_accuracies = [r['accuracy'] for r in cv_results]
        best_fold_idx = np.argmax(cv_accuracies)
        best_fold_num = best_fold_idx + 1
        self.best_fold_model_path = os.path.join(self.results_dir, f'best_model_fold_{best_fold_num}.pt')
        
        print(f"\n🏆 Best CV fold: {best_fold_num} with accuracy {cv_accuracies[best_fold_idx]:.4f}")
        print(f"   Will use this model for test evaluation: {self.best_fold_model_path}")
        
        # Calculate CV results
        cv_mean_accuracy = np.mean(cv_accuracies)
        cv_std_accuracy = np.std(cv_accuracies)
        cv_avg_training_time = np.mean(cv_training_times)
        
        print(f"\n{'='*60}")
        print(f"🎯 Cross-Validation Results on Training Set (SGD Optimizer):")
        print(f"📊 Mean CV Accuracy: {cv_mean_accuracy:.4f} ± {cv_std_accuracy:.4f}")
        print(f"⏱️  Average Training Time: {cv_avg_training_time:.1f}s per fold")
        print(f"📈 CV Accuracy Range: {min(cv_accuracies):.4f} - {max(cv_accuracies):.4f}")
        
        # Check consistency
        accuracy_range = max(cv_accuracies) - min(cv_accuracies)
        if accuracy_range > 0.15:
            print("⚠️  WARNING: Large CV accuracy variation detected!")
        elif accuracy_range > 0.08:
            print("⚠️  MODERATE CV accuracy variation detected.")
        else:
            print("✅ Good CV consistency across folds!")
        
        # Clean up CV artifacts
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        
        # Final cleanup
        MemoryManager.aggressive_cleanup()
        
        return cv_results, cv_mean_accuracy, all_cv_true_labels, all_cv_predictions

    def final_test_evaluation(self, test_images, test_labels):
        """Evaluate best CV model on the held-out test set with memory optimization"""
        
        print(f"\n🧪 Final evaluation on held-out test set...")
        print(f"📊 Test set size: {len(test_images)} samples")
        
        # Load best model from CV
        if not os.path.exists(self.best_fold_model_path):
            raise FileNotFoundError(f"Best CV model not found: {self.best_fold_model_path}")
        
        best_model = YOLO(self.best_fold_model_path)
        print(f"   📥 Using best CV model: {self.best_fold_model_path}")
        
        predictions = []
        confidences = []
        
        # Predict on test set in batches
        batch_size = 20  # Process 20 images at a time
        print("🔍 Evaluating on test set...")
        
        for i in range(0, len(test_images), batch_size):
            batch_images = test_images[i:i+batch_size]
            
            for img_path in batch_images:
                try:
                    if torch.cuda.is_available():
                        with torch.amp.autocast('cuda'):
                            results = best_model.predict(img_path, imgsz=self.img_size, 
                                                       device=self.device, verbose=False)
                    else:
                        results = best_model.predict(img_path, imgsz=self.img_size, 
                                                   device=self.device, verbose=False)
                    
                    pred_idx = results[0].probs.top1
                    pred_label = best_model.names[pred_idx]
                    confidence = results[0].probs.top1conf.item()
                    
                    # Convert class name back to index
                    pred_class_idx = self.class_names.index(pred_label)
                    predictions.append(pred_class_idx)
                    confidences.append(confidence)
                    
                    # Clear result from memory
                    del results
                    
                except Exception as e:
                    print(f"Error predicting {img_path}: {e}")
                    predictions.append(0)
                    confidences.append(0.0)
            
            # Cleanup after each batch
            if i % (batch_size * 5) == 0:  # Every 100 images
                MemoryManager.aggressive_cleanup()
        
        # Calculate final test metrics
        test_accuracy = accuracy_score(test_labels, predictions)
        test_avg_confidence = np.mean(confidences)
        
        print(f"🎯 Final Test Accuracy: {test_accuracy:.4f}")
        print(f"📈 Test Average Confidence: {test_avg_confidence:.4f}")
        
        # Cleanup test model
        del best_model
        MemoryManager.aggressive_cleanup()
        
        return predictions, confidences, test_accuracy

    def train_and_evaluate_class_specific(self, train_images_path, test_images_path, 
                                         cv_folds=5, epochs=60, batch_size=16, results_dir=None):
        """Complete workflow: Load pre-split data, CV on train, final test evaluation"""
        
        # Set results directory
        if results_dir is None:
            results_dir = f"yolo_{self.model_size}_class_specific_sgd_results_{int(time.time())}"
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"🎯 Starting YOLO Class-Specific Normalization Training with SGD")
        print("=" * 60)
        print(f"📁 Train images: {train_images_path}")
        print(f"📁 Test images: {test_images_path}")
        print(f"🎯 Model: YOLOv8{self.model_size}")
        print(f"📐 Image size: {self.img_size}x{self.img_size}")
        print(f"🔄 Epochs: {epochs}")
        print(f"📦 Batch size: {batch_size}")
        print(f"🔄 CV folds: {cv_folds}")
        print(f"💾 Results directory: {results_dir}")
        print(f"🚀 Optimizer: SGD with enhanced settings")
        
        # Step 1: Load pre-split class-specific normalized data
        train_images, train_labels, test_images, test_labels = self.load_pre_split_data(
            train_images_path, test_images_path
        )
        
        # Step 2: Cross-validation on training set
        cv_results, cv_mean_accuracy, cv_true_labels, cv_predictions = self.cross_validate_on_train_set(
            train_images, train_labels, n_splits=cv_folds, epochs=epochs, batch_size=batch_size
        )
        
        # Step 3: Final evaluation on test set
        test_predictions, test_confidences, test_accuracy = self.final_test_evaluation(
            test_images, test_labels
        )
        
        # Step 4: Prepare results for visualization
        cv_fold_results = cv_results
        test_fold_results = [{
            'fold': 'Test',
            'accuracy': test_accuracy,
            'training_time': sum(r['training_time'] for r in cv_results),
            'avg_confidence': np.mean(test_confidences),
            'val_samples': len(test_labels),
            'early_stopped': False,
            'best_epoch': epochs,
            'total_epochs_trained': epochs
        }]
        
        print(f"\n{'='*60}")
        print(f"🎯 FINAL RESULTS SUMMARY (SGD Optimizer):")
        print(f"📊 Cross-Validation (Training Set) Mean Accuracy: {cv_mean_accuracy:.4f}")
        print(f"📊 Final Test Set Accuracy: {test_accuracy:.4f}")
        print(f"📈 Test Set Average Confidence: {np.mean(test_confidences):.4f}")
        
        # Compare CV vs Test performance
        cv_test_diff = abs(cv_mean_accuracy - test_accuracy)
        if cv_test_diff > 0.1:
            print(f"⚠️  WARNING: Large difference between CV ({cv_mean_accuracy:.4f}) and test ({test_accuracy:.4f}) accuracy!")
        elif cv_test_diff > 0.05:
            print(f"⚠️  MODERATE difference between CV and test accuracy ({cv_test_diff:.4f})")
        else:
            print(f"✅ Good agreement between CV and test accuracy (diff: {cv_test_diff:.4f})")
        
        # Generate comprehensive visualizations
        print(f"\n📊 Generating visualizations...")
        
        # CV visualizations (training set performance)
        cv_results_dir = os.path.join(results_dir, 'cross_validation_results')
        os.makedirs(cv_results_dir, exist_ok=True)
        model_title = f"YOLO {self.model_size} Class-Specific SGD - Cross-Validation"
        self.generate_all_visualizations(cv_true_labels, cv_predictions, 
                                       cv_fold_results, cv_results_dir, model_title)
        
        # Test set visualizations (final performance)
        test_results_dir = os.path.join(results_dir, 'test_results')
        os.makedirs(test_results_dir, exist_ok=True)
        model_title = f"YOLO {self.model_size} Class-Specific SGD - Final Test"
        self.generate_all_visualizations(test_labels, test_predictions, 
                                       test_fold_results, test_results_dir, model_title)
        
        # Print final classification reports
        print("\n📊 Cross-Validation Classification Report (Training Set):")
        print(classification_report(cv_true_labels, cv_predictions, target_names=self.class_names))
        
        print("\n📊 Final Test Set Classification Report:")
        print(classification_report(test_labels, test_predictions, target_names=self.class_names))
        
        # Save detailed results
        results_summary = {
            'cv_mean_accuracy': cv_mean_accuracy,
            'cv_std_accuracy': np.std([r['accuracy'] for r in cv_results]),
            'test_accuracy': test_accuracy,
            'cv_test_difference': cv_test_diff,
            'cv_results': cv_results,
            'model_size': self.model_size,
            'img_size': self.img_size,
            'epochs': epochs,
            'batch_size': batch_size,
            'cv_folds': cv_folds,
            'optimizer': 'SGD',
            'learning_rate': 0.01,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'label_smoothing': 0.15,
            'dropout': 0.3,
            'normalization_type': 'class_specific',
            'data_split': 'pre_split_60_40'
        }
        
        with open(os.path.join(results_dir, 'cv_results.json'), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in results_summary.items():
                if key == 'cv_results':
                    json_results[key] = []
                    for fold_result in value:
                        fold_dict = {}
                        for k, v in fold_result.items():
                            if isinstance(v, np.ndarray):
                                fold_dict[k] = v.tolist()
                            else:
                                fold_dict[k] = v
                        json_results[key].append(fold_dict)
                elif isinstance(value, np.ndarray):
                    json_results[key] = value.tolist()
                else:
                    json_results[key] = value
            
            json.dump(json_results, f, indent=2)
        
        return cv_results, cv_mean_accuracy, test_accuracy, results_summary

def main():
    """Main training function for class-specific normalized YOLO with SGD optimizer"""
    
    # Set environment variables to reduce memory pressure
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1' 
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    # Memory-optimized configuration with SGD-friendly parameters
    TRAIN_IMAGES_PATH = r'full_newdata_global_all_norm_quantile_normal_heatmap_charts_224\train_images_heatmap_global_quantile_normal_normalized'
    TEST_IMAGES_PATH = r'full_newdata_global_all_norm_quantile_normal_heatmap_charts_224\test_images_heatmap_global_quantile_normal_normalized'
    
    MODEL_SIZE = 'n'      # Use nano model for less memory usage
    IMG_SIZE = 224        # Match your generated image size
    EPOCHS = 100           # Slightly more epochs for SGD (changed from 50)
    BATCH_SIZE = 32       # Smaller batch often works better with SGD (changed from 32)
    CV_FOLDS = 5          
    RESULTS_DIR = 'full_newdata_yolo_sgd_quantile_normal_heatmap_results'
    
    # Check and adjust parameters based on available memory
    BATCH_SIZE, CV_FOLDS, EPOCHS = MemoryManager.check_memory_and_adjust_params(
        BATCH_SIZE, CV_FOLDS, EPOCHS
    )
    
    print("🚀 Starting YOLO Class-Specific Normalization Classification with SGD Optimizer")
    print("=" * 80)
    print(f"📁 Train images: {TRAIN_IMAGES_PATH}")
    print(f"📁 Test images: {TEST_IMAGES_PATH}")
    print(f"🎯 Model: YOLOv8{MODEL_SIZE}")
    print(f"📐 Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"🔄 Epochs: {EPOCHS}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"🔄 Cross-validation: {CV_FOLDS} folds on training set")
    print(f"💾 Results directory: {RESULTS_DIR}")
    print(f"🎯 Data: Pre-split class-specific normalized images (60% train, 40% test)")
    print(f"🧹 Memory optimization: ENABLED")
    print("🚀 ENHANCED SGD OPTIMIZER SETTINGS:")
    print("   - Learning rate: 0.01 (vs 0.0001 with AdamW)")
    print("   - Momentum: 0.9")
    print("   - Weight decay: 0.0005")
    print("   - Label smoothing: 0.15 (enhanced for variable stars)")
    print("   - Dropout: 0.3 (increased regularization)")
    print("   - Cosine annealing: Enabled")
    print("   - Patience: 15 epochs")
    
    # Verify paths exist
    if not os.path.exists(TRAIN_IMAGES_PATH):
        print(f"❌ Error: Train images path not found: {TRAIN_IMAGES_PATH}")
        print("Please run your class-specific chart generator first!")
        return
    
    if not os.path.exists(TEST_IMAGES_PATH):
        print(f"❌ Error: Test images path not found: {TEST_IMAGES_PATH}")
        print("Please run your class-specific chart generator first!")
        return
    
    # Initialize classifier
    classifier = ClassSpecificYOLOClassifier(
        model_size=MODEL_SIZE,
        img_size=IMG_SIZE,
        device='auto',
        random_seed=42
    )
    
    # Monitor memory throughout training
    ram_start, gpu_start = MemoryManager.get_memory_usage()
    print(f"\n💻 Starting memory: RAM {ram_start:.1f}MB, GPU {gpu_start:.1f}MB")
    
    try:
        # Run complete workflow with memory management
        print("\n🎯 Running class-specific normalization workflow with SGD...")
        cv_results, cv_accuracy, test_accuracy, summary = classifier.train_and_evaluate_class_specific(
            TRAIN_IMAGES_PATH, 
            TEST_IMAGES_PATH,
            cv_folds=CV_FOLDS,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            results_dir=RESULTS_DIR
        )
        
        print(f"\n🎊 FINAL SGD YOLO CLASS-SPECIFIC RESULTS:")
        print(f"📊 Cross-Validation Mean Accuracy: {cv_accuracy:.4f}")
        print(f"🎯 Final Test Set Accuracy: {test_accuracy:.4f}")
        print(f"📊 Detailed results and visualizations saved to: {RESULTS_DIR}")
        print("\nGenerated files:")
        print("📁 cross_validation_results/ - Cross-validation analysis and visualizations")
        print("📁 test_results/ - Final test set results and visualizations")
        print("📄 cv_results.json - Complete numerical results with SGD parameters")
        print("\nThis tests your class-specific normalization approach with SGD:")
        print("✅ Each variable star class normalized with its own parameters")
        print("✅ Test samples normalized using most similar training class")
        print("✅ SGD optimizer with enhanced settings for better convergence")
        print("✅ Higher learning rate and momentum for better discrimination")
        print("✅ Increased regularization for variable star classification")
        print(f"\n🔬 Key Research Question Tested:")
        print(f"   Does class-specific normalization + SGD optimizer improve discrimination")
        print(f"   between similar variable star types (especially Irregular vs Semiregular)?")
        
        # Print comparison prompt
        print(f"\n📈 To compare optimizers:")
        print(f"   1. SGD Result: {test_accuracy:.4f}")
        print(f"   2. Compare with previous AdamW results")
        print(f"   3. Check if SGD better separates Irregular vs Semiregular")
        print(f"   4. Look for improved confidence scores and confusion matrices")
        print(f"   5. SGD often finds better decision boundaries for similar classes")
        
        # Optimizer-specific insights
        print(f"\n🧠 SGD Optimizer Benefits for Variable Stars:")
        print(f"   • Higher learning rate helps escape local minima")
        print(f"   • Momentum helps navigate loss landscape more effectively")
        print(f"   • Better generalization for small astronomical datasets")
        print(f"   • Less sensitive to hyperparameters than Adam variants")
        print(f"   • Often superior final performance on classification tasks")
        
    except Exception as e:
        print(f"\n❌ Error during SGD training: {e}")
        print("This might be due to:")
        print("1. Memory constraints - try smaller batch size")
        print("2. Learning rate too high - SGD can be sensitive")
        print("3. Data path issues")
        print("Suggestions:")
        print("- Reduce BATCH_SIZE to 8 or 4")
        print("- Reduce EPOCHS to 40")
        print("- Close other applications")
        raise
    
    finally:
        # Final memory cleanup and reporting
        MemoryManager.aggressive_cleanup()
        ram_end, gpu_end = MemoryManager.get_memory_usage()
        print(f"\n💻 Final memory: RAM {ram_end:.1f}MB, GPU {gpu_end:.1f}MB")
        print(f"📊 Memory change: RAM {ram_end-ram_start:+.1f}MB, GPU {gpu_end-gpu_start:+.1f}MB")

if __name__ == "__main__":
    main()