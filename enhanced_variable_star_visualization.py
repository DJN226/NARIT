"""
Enhanced Variable Star Classification Visualization Mixin
Specialized for 9-class variable star classification with additional astronomy-specific visualizations
FIXED VERSION - Split classification report and improved color scheme
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.patches as mpatches

class VariableStarVisualizationMixin:
    """
    Enhanced mixin class for variable star classification visualization.
    Includes astronomy-specific visualizations and improved general metrics.
    """
    
    def generate_confusion_matrix(self, y_true, y_pred, save_path, model_title="Model"):
        """Generate and save confusion matrix visualization with astronomy styling - FIXED VERSION"""
        print("📊 Generating confusion matrix...")
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(self.class_names))))
        
        # Create figure with appropriate size for 9 classes and proper spacing
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))  # Increased figure size
        
        # Raw confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=self.class_names, yticklabels=self.class_names,
                cbar_kws={'label': 'Number of Samples'}, ax=ax1)
        
        ax1.set_title(f'{model_title}\nConfusion Matrix (Raw Counts)', 
                    fontsize=14, fontweight='bold', pad=20)  # Added padding
        ax1.set_xlabel('Predicted Variable Star Type', fontsize=12)
        ax1.set_ylabel('True Variable Star Type', fontsize=12)
        ax1.tick_params(axis='x', rotation=45, labelsize=10)
        ax1.tick_params(axis='y', rotation=0, labelsize=10)
        
        # Normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=self.class_names, yticklabels=self.class_names,
                cbar_kws={'label': 'Proportion'}, ax=ax2, vmin=0, vmax=1)
        
        ax2.set_title(f'{model_title}\nConfusion Matrix (Normalized)', 
                    fontsize=14, fontweight='bold', pad=20)  # Added padding
        ax2.set_xlabel('Predicted Variable Star Type', fontsize=12)
        ax2.set_ylabel('True Variable Star Type', fontsize=12)
        ax2.tick_params(axis='x', rotation=45, labelsize=10)
        ax2.tick_params(axis='y', rotation=0, labelsize=10)
        
        # Add overall accuracy
        overall_accuracy = np.trace(cm) / np.sum(cm)
        
        # Fixed: Use suptitle with proper positioning and spacing
        fig.suptitle(f'Variable Star Classification Results\nOverall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)', 
                    fontsize=16, fontweight='bold', y=0.95)  # Moved down from 0.98 to 0.95
        
        # Adjust layout with more space at top
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # More space at top (was 0.88)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Confusion matrices saved to: {save_path}")
    
    def generate_classification_report_heatmap(self, y_true, y_pred, save_path, model_title="Model"):
        """Generate classification report heatmap (metrics only) with white-to-blue color scheme"""
        print("📈 Generating classification report heatmap...")
        
        # Generate classification report as dictionary
        report = classification_report(y_true, y_pred, target_names=self.class_names, 
                                     output_dict=True, zero_division=0)
        
        # Create DataFrame for heatmap (metrics only)
        metrics_data = []
        class_labels = []
        
        # Add individual classes
        for class_name in self.class_names:
            if class_name in report:
                metrics_data.append([
                    report[class_name]['precision'],
                    report[class_name]['recall'], 
                    report[class_name]['f1-score']
                ])
                class_labels.append(class_name)
        
        # Add overall metrics
        metrics_data.append([
            report['accuracy'],
            report['accuracy'], 
            report['accuracy']
        ])
        class_labels.append('Overall Accuracy')
        
        metrics_data.append([
            report['macro avg']['precision'],
            report['macro avg']['recall'],
            report['macro avg']['f1-score']
        ])
        class_labels.append('Macro Average')
        
        metrics_data.append([
            report['weighted avg']['precision'],
            report['weighted avg']['recall'],
            report['weighted avg']['f1-score']
        ])
        class_labels.append('Weighted Average')
        
        # Create DataFrame (metrics only)
        df_metrics = pd.DataFrame(metrics_data, 
                                columns=['Precision', 'Recall', 'F1-Score'],
                                index=class_labels)
        
        # Create single figure for metrics heatmap
        fig, ax = plt.subplots(1, 1, figsize=(10, 12))
        
        # Metrics heatmap with white-to-blue color scheme
        sns.heatmap(df_metrics, annot=True, fmt='.3f', cmap='Blues',
                   cbar_kws={'label': 'Score'}, vmin=0.0, vmax=1.0,
                   linewidths=0.5, linecolor='white', ax=ax)
        
        ax.set_title(f'{model_title}\nClassification Metrics by Variable Star Type', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Performance Metrics', fontsize=12)
        ax.set_ylabel('Variable Star Classes', fontsize=12)
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='x', labelsize=11)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Classification report heatmap saved to: {save_path}")
    
    def generate_sample_distribution(self, y_true, y_pred, save_path, model_title="Model"):
        """Generate sample distribution visualization as separate file"""
        print("📊 Generating sample distribution chart...")
        
        # Generate classification report to get support values
        report = classification_report(y_true, y_pred, target_names=self.class_names, 
                                     output_dict=True, zero_division=0)
        
        # Create figure for sample distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 1. Support bar chart (horizontal)
        support_data = []
        class_labels = []
        for class_name in self.class_names:
            if class_name in report:
                support_data.append(report[class_name]['support'])
                class_labels.append(class_name)
        
        # Create color gradient from white to blue
        colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(class_labels)))
        
        bars = ax1.barh(class_labels, support_data, 
                       color=colors, alpha=0.8, edgecolor='navy', linewidth=1)
        ax1.set_xlabel('Number of Test Samples', fontsize=12)
        ax1.set_title('Sample Distribution\nper Variable Star Type', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, support_data)):
            ax1.text(bar.get_width() + max(support_data)*0.01, bar.get_y() + bar.get_height()/2,
                    f'{int(value)}', ha='left', va='center', fontweight='bold', fontsize=10)
        
        # 2. Percentage pie chart
        wedges, texts, autotexts = ax2.pie(support_data, labels=class_labels, autopct='%1.1f%%',
                                          colors=colors, startangle=90)
        ax2.set_title('Test Set Distribution\nby Variable Star Type (%)', fontsize=14, fontweight='bold')
        
        # Make percentage text bold and visible
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        # Adjust text size for labels
        for text in texts:
            text.set_fontsize(9)
        
        # Add overall statistics
        total_samples = sum(support_data)
        mean_samples = np.mean(support_data)
        std_samples = np.std(support_data)
        
        fig.suptitle(f'{model_title} - Sample Distribution Analysis\n'
                    f'Total: {total_samples} samples | Mean: {mean_samples:.1f} ± {std_samples:.1f} per class', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Sample distribution chart saved to: {save_path}")
    
    def generate_performance_analysis(self, y_true, y_pred, fold_results, save_path, model_title="Model"):
        """Generate comprehensive performance analysis for variable star classification"""
        print("📊 Generating variable star performance analysis...")
        
        # Generate classification report
        report = classification_report(y_true, y_pred, target_names=self.class_names, 
                                     output_dict=True, zero_division=0)
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 15))
        
        # Create a 3x3 grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Fold accuracy evolution with early stopping info
        ax1 = fig.add_subplot(gs[0, 0])
        fold_accuracies = [r['accuracy'] for r in fold_results]
        best_epochs = [r.get('best_epoch', 0) for r in fold_results]
        early_stopped = [r.get('early_stopped', False) for r in fold_results]
        
        # Color bars based on early stopping
        colors = ['red' if stopped else 'steelblue' for stopped in early_stopped]
        
        bars = ax1.bar(range(1, len(fold_results)+1), fold_accuracies, 
                      color=colors, alpha=0.7, edgecolor='black')
        ax1.axhline(y=np.mean(fold_accuracies), color='green', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(fold_accuracies):.3f}')
        ax1.set_title('Cross-Validation Accuracy\n(Red = Early Stopped)', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Fold Number')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add value labels with epoch info
        for i, (bar, acc, epoch, stopped) in enumerate(zip(bars, fold_accuracies, best_epochs, early_stopped)):
            label = f"{acc:.3f}\n(E:{epoch})" if epoch > 0 else f"{acc:.3f}"
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    label, ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 2. Training time analysis
        ax2 = fig.add_subplot(gs[0, 1])
        training_times = [r['training_time']/60 for r in fold_results]
        bars = ax2.bar(range(1, len(fold_results)+1), training_times,
                      color='orange', alpha=0.7, edgecolor='black')
        ax2.set_title('Training Time\nper Fold', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Fold Number')
        ax2.set_ylabel('Time (minutes)')
        ax2.grid(True, alpha=0.3)
        
        for i, (bar, time_val) in enumerate(zip(bars, training_times)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(training_times)*0.01,
                    f"{time_val:.1f}m", ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 3. Confidence distribution
        ax3 = fig.add_subplot(gs[0, 2])
        all_confidences = []
        for fold_result in fold_results:
            if 'avg_confidence' in fold_result:
                all_confidences.append(fold_result['avg_confidence'])
        if all_confidences:
            bars = ax3.bar(range(1, len(all_confidences)+1), all_confidences,
                          color='green', alpha=0.7, edgecolor='black')
            ax3.set_title('Average Prediction\nConfidence by Fold', fontweight='bold', fontsize=12)
            ax3.set_xlabel('Fold Number')
            ax3.set_ylabel('Confidence')
            ax3.set_ylim(0, 1)
            ax3.grid(True, alpha=0.3)
            
            for i, (bar, conf) in enumerate(zip(bars, all_confidences)):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{conf:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 4. Per-class precision (horizontal bar chart) - using Blues colormap
        ax4 = fig.add_subplot(gs[1, :])
        precision_scores = [report[class_name]['precision'] for class_name in self.class_names 
                           if class_name in report]
        
        # Create a white-to-blue color map
        colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(self.class_names)))
        
        bars = ax4.barh(self.class_names, precision_scores, color=colors, alpha=0.8, 
                       edgecolor='navy', linewidth=1)
        ax4.set_title('Precision by Variable Star Type', fontweight='bold', fontsize=14)
        ax4.set_xlabel('Precision Score')
        ax4.set_xlim(0, 1)
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, precision in zip(bars, precision_scores):
            ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{precision:.3f}", ha='left', va='center', fontweight='bold', fontsize=10)
        
        # 5. Per-class recall (horizontal bar chart) - using Blues colormap
        ax5 = fig.add_subplot(gs[2, 0])
        recall_scores = [report[class_name]['recall'] for class_name in self.class_names 
                        if class_name in report]
        
        bars = ax5.barh(self.class_names, recall_scores, color=colors, alpha=0.8, 
                       edgecolor='navy', linewidth=1)
        ax5.set_title('Recall by Variable Star Type', fontweight='bold', fontsize=12)
        ax5.set_xlabel('Recall Score')
        ax5.set_xlim(0, 1)
        ax5.grid(True, alpha=0.3, axis='x')
        
        for bar, recall in zip(bars, recall_scores):
            ax5.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{recall:.3f}", ha='left', va='center', fontweight='bold', fontsize=10)
        
        # 6. Per-class F1-score (horizontal bar chart) - using Blues colormap
        ax6 = fig.add_subplot(gs[2, 1])
        f1_scores = [report[class_name]['f1-score'] for class_name in self.class_names 
                    if class_name in report]
        
        bars = ax6.barh(self.class_names, f1_scores, color=colors, alpha=0.8, 
                       edgecolor='navy', linewidth=1)
        ax6.set_title('F1-Score by Variable Star Type', fontweight='bold', fontsize=12)
        ax6.set_xlabel('F1-Score')
        ax6.set_xlim(0, 1)
        ax6.grid(True, alpha=0.3, axis='x')
        
        for bar, f1 in zip(bars, f1_scores):
            ax6.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{f1:.3f}", ha='left', va='center', fontweight='bold', fontsize=10)
        
        # 7. Class balance visualization
        ax7 = fig.add_subplot(gs[2, 2])
        class_counts = [list(y_true).count(i) for i in range(len(self.class_names))]
        
        # Calculate balance score (closer to 1 = more balanced)
        ideal_count = len(y_true) / len(self.class_names)
        balance_scores = [1 - abs(count - ideal_count) / ideal_count for count in class_counts]
        
        bars = ax7.bar(range(len(self.class_names)), balance_scores, 
                      color=colors, alpha=0.8, edgecolor='navy', linewidth=1)
        ax7.set_title('Class Balance Score\n(1 = Perfect Balance)', fontweight='bold', fontsize=12)
        ax7.set_xlabel('Variable Star Type')
        ax7.set_ylabel('Balance Score')
        ax7.set_xticks(range(len(self.class_names)))
        ax7.set_xticklabels(self.class_names, rotation=45, ha='right', fontsize=9)
        ax7.set_ylim(0, 1)
        ax7.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, balance_scores)):
            ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{score:.2f}", ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.suptitle(f'{model_title} - Comprehensive Performance Analysis', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Comprehensive performance analysis saved to: {save_path}")
    
    def generate_variable_star_specific_analysis(self, y_true, y_pred, save_path, model_title="Model"):
        """Generate variable star specific analysis including misclassification patterns"""
        print("🌟 Generating variable star specific analysis...")
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(self.class_names))))
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Misclassification heatmap (off-diagonal elements only)
        misclass_matrix = cm.copy()
        np.fill_diagonal(misclass_matrix, 0)  # Remove correct classifications
        
        mask = misclass_matrix == 0
        sns.heatmap(misclass_matrix, annot=True, fmt='d', cmap='Reds',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   mask=mask, cbar_kws={'label': 'Misclassifications'}, ax=ax1)
        
        ax1.set_title('Variable Star Misclassification Patterns\n(Most Common Confusions)', 
                     fontweight='bold', fontsize=14)
        ax1.set_xlabel('Predicted Variable Star Type')
        ax1.set_ylabel('True Variable Star Type')
        ax1.tick_params(axis='x', rotation=45, labelsize=10)
        ax1.tick_params(axis='y', rotation=0, labelsize=10)
        
        # 2. Per-class accuracy radar chart
        per_class_accuracy = []
        for i in range(len(self.class_names)):
            if cm[i].sum() > 0:
                acc = cm[i, i] / cm[i].sum()
            else:
                acc = 0
            per_class_accuracy.append(acc)
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(self.class_names), endpoint=False).tolist()
        per_class_accuracy += per_class_accuracy[:1]  # Complete the circle
        angles += angles[:1]
        
        ax2.plot(angles, per_class_accuracy, 'o-', linewidth=2, color='blue')
        ax2.fill(angles, per_class_accuracy, alpha=0.25, color='blue')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(self.class_names, fontsize=10)
        ax2.set_ylim(0, 1)
        ax2.set_title('Per-Class Accuracy\n(Radar Chart)', fontweight='bold', fontsize=14)
        ax2.grid(True)
        
        # Add accuracy values as text
        for angle, acc, class_name in zip(angles[:-1], per_class_accuracy[:-1], self.class_names):
            ax2.text(angle, acc + 0.05, f'{acc:.2f}', ha='center', va='center', 
                    fontweight='bold', fontsize=9)
        
        # 3. Most confused pairs analysis
        # Find top misclassification pairs
        confused_pairs = []
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append((self.class_names[i], self.class_names[j], cm[i, j]))
        
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        top_confusions = confused_pairs[:8]  # Top 8 confusions
        
        if top_confusions:
            confusion_labels = [f"{pair[0]}\n→ {pair[1]}" for pair in top_confusions]
            confusion_counts = [pair[2] for pair in top_confusions]
            
            bars = ax3.bar(range(len(confusion_labels)), confusion_counts, 
                          color='coral', alpha=0.7, edgecolor='black')
            ax3.set_title('Most Common Variable Star\nMisclassification Pairs', 
                         fontweight='bold', fontsize=14)
            ax3.set_xlabel('True → Predicted')
            ax3.set_ylabel('Number of Misclassifications')
            ax3.set_xticks(range(len(confusion_labels)))
            ax3.set_xticklabels(confusion_labels, rotation=45, ha='right', fontsize=9)
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, count in zip(bars, confusion_counts):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(confusion_counts)*0.01,
                        f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 4. Classification difficulty analysis
        class_difficulties = []
        for i, class_name in enumerate(self.class_names):
            total_predictions_for_class = cm[:, i].sum()  # Total predicted as this class
            correct_predictions = cm[i, i]
            if total_predictions_for_class > 0:
                precision = correct_predictions / total_predictions_for_class
                recall = correct_predictions / cm[i].sum() if cm[i].sum() > 0 else 0
                difficulty_score = 1 - ((precision + recall) / 2)  # Higher = more difficult
            else:
                difficulty_score = 1
            class_difficulties.append((class_name, difficulty_score))
        
        class_difficulties.sort(key=lambda x: x[1], reverse=True)
        
        difficulty_names = [item[0] for item in class_difficulties]
        difficulty_scores = [item[1] for item in class_difficulties]
        
        # Color code by difficulty
        colors = ['red' if score > 0.5 else 'orange' if score > 0.3 else 'green' 
                 for score in difficulty_scores]
        
        bars = ax4.barh(difficulty_names, difficulty_scores, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_title('Variable Star Classification\nDifficulty Ranking', fontweight='bold', fontsize=14)
        ax4.set_xlabel('Difficulty Score (1 = Most Difficult)')
        ax4.set_xlim(0, 1)
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, score in zip(bars, difficulty_scores):
            ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{score:.3f}", ha='left', va='center', fontweight='bold', fontsize=10)
        
        # Add legend for difficulty colors
        red_patch = mpatches.Patch(color='red', alpha=0.7, label='High Difficulty (>0.5)')
        orange_patch = mpatches.Patch(color='orange', alpha=0.7, label='Medium Difficulty (0.3-0.5)')
        green_patch = mpatches.Patch(color='green', alpha=0.7, label='Low Difficulty (<0.3)')
        ax4.legend(handles=[red_patch, orange_patch, green_patch], loc='lower right')
        
        plt.suptitle(f'{model_title} - Variable Star Specific Analysis', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Variable star specific analysis saved to: {save_path}")
    
    def generate_all_visualizations(self, y_true, y_pred, fold_results, results_dir, model_title="Model"):
        """Generate all enhanced visualizations for variable star classification - UPDATED"""
        print(f"\n📊 Generating all enhanced {model_title} visualizations...")
        
        os.makedirs(results_dir, exist_ok=True)
        
        # 1. Enhanced Confusion Matrix
        cm_path = os.path.join(results_dir, 'confusion_matrix.png')
        self.generate_confusion_matrix(y_true, y_pred, cm_path, model_title)
        
        # 2. Classification Report Heatmap (metrics only) - NEW
        cr_heatmap_path = os.path.join(results_dir, 'classification_report.png')
        self.generate_classification_report_heatmap(y_true, y_pred, cr_heatmap_path, model_title)
        
        # 3. Sample Distribution (separate file) - NEW
        sample_dist_path = os.path.join(results_dir, 'sample_distribution.png')
        self.generate_sample_distribution(y_true, y_pred, sample_dist_path, model_title)
        
        # 4. Comprehensive Performance Analysis
        perf_path = os.path.join(results_dir, 'performance_analysis.png')
        self.generate_performance_analysis(y_true, y_pred, fold_results, perf_path, model_title)
        
        # 5. Variable Star Specific Analysis
        vs_analysis_path = os.path.join(results_dir, 'variable_star_analysis.png')
        self.generate_variable_star_specific_analysis(y_true, y_pred, vs_analysis_path, model_title)
        
        # 6. Enhanced Text Classification Report
        from sklearn.metrics import classification_report
        report_text = classification_report(y_true, y_pred, target_names=self.class_names)
        with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
            f.write(f"{model_title} - Variable Star Classification Report\n")
            f.write("=" * 80 + "\n\n")
            f.write("DATASET INFORMATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total test samples: {len(y_true)}\n")
            f.write(f"Number of variable star types: {len(self.class_names)}\n")
            f.write(f"Variable star types: {', '.join(self.class_names)}\n\n")
            
            f.write("CROSS-VALIDATION RESULTS:\n")
            f.write("-" * 40 + "\n")
            f.write("Fold-wise Results:\n")
            for i, result in enumerate(fold_results):
                early_stop_info = ""
                if result.get('early_stopped', False):
                    early_stop_info = f" (Early Stopped at epoch {result.get('best_epoch', 'N/A')})"
                else:
                    early_stop_info = f" (Completed {result.get('total_epochs_trained', 'N/A')} epochs, best: {result.get('best_epoch', 'N/A')})"
                
                f.write(f"Fold {i+1}: {result['accuracy']:.4f} accuracy, "
                       f"{result.get('training_time', 0):.1f}s training time")
                if 'avg_confidence' in result:
                    f.write(f", {result['avg_confidence']:.4f} avg confidence")
                f.write(early_stop_info + "\n")
            
            # Add summary statistics
            accuracies = [r['accuracy'] for r in fold_results]
            early_stopped_count = sum([r.get('early_stopped', False) for r in fold_results])
            avg_best_epoch = np.mean([r.get('best_epoch', 0) for r in fold_results])
            
            f.write(f"\nSummary Statistics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}\n")
            f.write(f"Min Accuracy: {np.min(accuracies):.4f}\n")
            f.write(f"Max Accuracy: {np.max(accuracies):.4f}\n")
            f.write(f"Total Training Time: {sum([r.get('training_time', 0) for r in fold_results]):.1f}s "
                   f"({sum([r.get('training_time', 0) for r in fold_results])/60:.1f} minutes)\n")
            f.write(f"Early Stopped Folds: {early_stopped_count}/5\n")
            f.write(f"Average Best Epoch: {avg_best_epoch:.1f}\n\n")
            
            f.write("DETAILED CLASSIFICATION METRICS:\n")
            f.write("-" * 40 + "\n")
            f.write(report_text)
            
            # Add class distribution
            f.write("\n\nCLASS DISTRIBUTION IN TEST SET:\n")
            f.write("-" * 40 + "\n")
            unique, counts = np.unique(y_true, return_counts=True)
            for class_idx, count in zip(unique, counts):
                f.write(f"{self.class_names[class_idx]}: {count} samples ({count/len(y_true)*100:.1f}%)\n")
        
        print(f"📁 All enhanced {model_title} visualizations saved to: {results_dir}")
        print("Generated files:")
        print("📈 confusion_matrix.png - Enhanced confusion matrices (raw + normalized)")
        print("🎨 classification_report.png - Classification metrics heatmap (white-to-blue colors)")
        print("📊 sample_distribution.png - Sample distribution bar chart and pie chart (separate file)")
        print("📊 performance_analysis.png - Comprehensive performance analysis (7 charts)")
        print("🌟 variable_star_analysis.png - Variable star specific analysis (misclassifications, difficulty)")
        print("📄 classification_report.txt - Comprehensive text report with cross-validation stats")
        
        return results_dir