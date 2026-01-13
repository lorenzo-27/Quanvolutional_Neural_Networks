"""
Visualization utilities for QNN experiments.
Generates feature maps, training curves, and analysis plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Tuple
from sklearn.metrics import confusion_matrix
import wandb

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


class Visualizer:
    """Handles all visualization tasks for QNN experiments."""

    def __init__(self, save_dir: str = "./results", use_wandb: bool = True):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb

    def plot_feature_maps(
            self,
            original_images: np.ndarray,
            quantum_features: np.ndarray,
            classical_features: Optional[np.ndarray] = None,
            n_samples: int = 8,
            save_name: str = "feature_maps.png"
    ):
        """
        Visualize original images vs quantum/classical feature maps.

        Args:
            original_images: Original input images [N, H, W]
            quantum_features: Quantum-processed features [N, H', W', C]
            classical_features: Classical features (optional) [N, H', W', C]
            n_samples: Number of samples to show
            save_name: Filename to save
        """
        n_samples = min(n_samples, len(original_images))

        # Determine number of rows
        n_rows = 1 + quantum_features.shape[-1]  # Input + quantum channels
        if classical_features is not None:
            n_rows += classical_features.shape[-1]

        fig, axes = plt.subplots(
            n_rows, n_samples,
            figsize=(n_samples * 1.5, n_rows * 1.5)
        )

        if n_samples == 1:
            axes = axes[:, np.newaxis]

        for col in range(n_samples):
            # Original image
            axes[0, col].imshow(original_images[col], cmap='gray')
            axes[0, col].axis('off')
            if col == 0:
                axes[0, col].set_ylabel("Input", fontsize=10, rotation=0, ha='right')

            # Quantum feature maps
            n_quantum_ch = quantum_features.shape[-1]
            for ch in range(n_quantum_ch):
                row = 1 + ch
                axes[row, col].imshow(
                    quantum_features[col, :, :, ch],
                    cmap='gray'
                )
                axes[row, col].axis('off')
                if col == 0:
                    axes[row, col].set_ylabel(
                        f"Quanv [{ch}]",
                        fontsize=9,
                        rotation=0,
                        ha='right'
                    )

            # Classical feature maps (if provided)
            if classical_features is not None:
                n_classical_ch = classical_features.shape[-1]
                for ch in range(n_classical_ch):
                    row = 1 + n_quantum_ch + ch
                    axes[row, col].imshow(
                        classical_features[col, :, :, ch],
                        cmap='gray'
                    )
                    axes[row, col].axis('off')
                    if col == 0:
                        axes[row, col].set_ylabel(
                            f"Classical [{ch}]",
                            fontsize=9,
                            rotation=0,
                            ha='right'
                        )

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        # Log to wandb
        if self.use_wandb:
            wandb.log({"feature_maps": wandb.Image(str(save_path))})

        return save_path

    def plot_training_curves(
            self,
            metrics: dict,
            save_name: str = "training_curves.png"
    ):
        """
        Plot training and validation metrics over time.

        Args:
            metrics: Dict with keys like 'qnn', 'classical', each containing
                    train_losses, train_accs, val_losses, val_accs
            save_name: Filename to save
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Colors for different models
        colors = {
            'qnn': '#2E86AB',
            'classical': '#A23B72',
            'random': '#F18F01'
        }

        # Plot losses
        ax = axes[0]
        for model_name, model_metrics in metrics.items():
            epochs = range(1, len(model_metrics['train_losses']) + 1)
            color = colors.get(model_name, 'gray')

            ax.plot(
                epochs,
                model_metrics['train_losses'],
                label=f'{model_name.upper()} Train',
                color=color,
                linestyle='--',
                alpha=0.7
            )
            ax.plot(
                epochs,
                model_metrics['val_losses'],
                label=f'{model_name.upper()} Val',
                color=color,
                linewidth=2
            )

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot accuracies
        ax = axes[1]
        for model_name, model_metrics in metrics.items():
            epochs = range(1, len(model_metrics['train_accs']) + 1)
            color = colors.get(model_name, 'gray')

            ax.plot(
                epochs,
                model_metrics['train_accs'],
                label=f'{model_name.upper()} Train',
                color=color,
                linestyle='--',
                alpha=0.7
            )
            ax.plot(
                epochs,
                model_metrics['val_accs'],
                label=f'{model_name.upper()} Val',
                color=color,
                linewidth=2
            )

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        # Log to wandb
        if self.use_wandb:
            wandb.log({"training_curves": wandb.Image(str(save_path))})

        return save_path

    def plot_confusion_matrix(
            self,
            predictions: np.ndarray,
            true_labels: np.ndarray,
            class_names: Optional[List[str]] = None,
            save_name: str = "confusion_matrix.png"
    ):
        """
        Plot confusion matrix.

        Args:
            predictions: Predicted labels
            true_labels: True labels
            class_names: Names of classes (optional)
            save_name: Filename to save
        """
        cm = confusion_matrix(true_labels, predictions)

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax,
            cbar_kws={'label': 'Count'}
        )

        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

        if class_names:
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        # Log to wandb
        if self.use_wandb:
            wandb.log({"confusion_matrix": wandb.Image(str(save_path))})

        return save_path

    def plot_quantum_channels(
            self,
            quantum_features: np.ndarray,
            sample_idx: int = 0,
            save_name: str = "quantum_channels.png"
    ):
        """
        Visualize all channels from quantum convolution for one sample.

        Args:
            quantum_features: Quantum features [N, H, W, C]
            sample_idx: Which sample to visualize
            save_name: Filename to save
        """
        n_channels = quantum_features.shape[-1]
        n_cols = min(8, n_channels)
        n_rows = (n_channels + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 2, n_rows * 2)
        )

        if n_rows == 1:
            axes = axes[np.newaxis, :]

        for ch in range(n_channels):
            row = ch // n_cols
            col = ch % n_cols

            axes[row, col].imshow(
                quantum_features[sample_idx, :, :, ch],
                cmap='viridis'
            )
            axes[row, col].set_title(f'Channel {ch}', fontsize=9)
            axes[row, col].axis('off')

        # Hide unused subplots
        for ch in range(n_channels, n_rows * n_cols):
            row = ch // n_cols
            col = ch % n_cols
            axes[row, col].axis('off')

        plt.suptitle(
            f'Quantum Feature Maps (Sample {sample_idx})',
            fontsize=14,
            fontweight='bold'
        )
        plt.tight_layout()

        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        # Log to wandb
        if self.use_wandb:
            wandb.log({"quantum_channels": wandb.Image(str(save_path))})

        return save_path

    def plot_comparison_bar(
            self,
            results: dict,
            save_name: str = "model_comparison.png"
    ):
        """
        Bar plot comparing final accuracies of different models.

        Args:
            results: Dict like {'QNN': 97.5, 'Classical': 96.2, ...}
            save_name: Filename to save
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        models = list(results.keys())
        accuracies = list(results.values())

        colors = ['#2E86AB', '#A23B72', '#F18F01'][:len(models)]

        bars = ax.bar(models, accuracies, color=colors, alpha=0.8)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.2f}%',
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold'
            )

        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 105])
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        # Log to wandb
        if self.use_wandb:
            wandb.log({"model_comparison": wandb.Image(str(save_path))})

        return save_path