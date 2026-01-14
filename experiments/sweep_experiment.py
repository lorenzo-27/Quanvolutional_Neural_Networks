"""
Sweep experiments for QNN research.
Performs grid search over models, encodings, circuits, and noise settings.
Generates comparative analysis plots.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import yaml
import json
import itertools
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.data.dataset import DatasetManager, preprocess_for_quantum
from src.quantum.quanvolution import QuanvolutionalLayer
from src.models.qnn import get_model
from src.training.trainer import Trainer
from src.utils.helpers import set_seed, get_device

console = Console()

# Sweep configuration
SWEEP_CONFIG = {
    "models": ["qnn", "classical", "random_nonlinear"],
    "encodings": ["threshold", "angle", "amplitude", "dense"],
    "circuits": ["random", "structured", "hardware_efficient"],
    "noise": [False, True],
}

# Base configuration template
BASE_CONFIG = {
    "experiment": {
        "name": "sweep_experiment",
        "seed": 42,
        "device": "auto"
    },
    "dataset": {
        "name": "MNIST",
        "path": "./data",
        "train_size": 500,
        "test_size": 100,
        "normalize": True
    },
    "quantum": {
        "n_qubits": 4,
        "kernel_size": 2,
        "stride": 2,
        "encoding": {
            "type": "angle",
            "threshold": 0.0,
            "scale_factor": 3.14159
        },
        "circuit": {
            "type": "hardware_efficient",
            "n_layers": 1,
            "gate_set": ["RX", "RY", "RZ", "CNOT"],
            "entanglement": "full",
            "random_seed": 0
        },
        "noise": {
            "enabled": False,
            "backend": "qasm_simulator",
            "noise_model": "fake_jakarta"
        }
    },
    "model": {
        "type": "qnn",
        "quanv_filters": 4,
        "conv1_filters": 32,
        "conv1_kernel": 3,
        "conv2_filters": 64,
        "conv2_kernel": 3,
        "fc1_units": 128,
        "dropout": 0.3,
        "num_classes": 10
    },
    "training": {
        "epochs": 10,
        "batch_size": 64,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "checkpoint_dir": "./checkpoints/sweep",
        "save_frequency": 5
    },
    "preprocessing": {
        "cache_dir": "./cache/sweep",
        "force_recompute": False
    },
    "visualization": {
        "save_dir": "./results/sweep"
    },
    "wandb": {
        "enabled": True
    }
}


class SweepExperiment:
    """Manages sweep experiments across different configurations."""

    def __init__(
            self,
            base_config: dict,
            sweep_config: dict,
            results_dir: str = "./results/sweep"
    ):
        self.base_config = base_config.copy()
        self.sweep_config = sweep_config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Cache for quantum preprocessing
        self.quantum_cache = {}

    def _create_config(
            self,
            model_type: str,
            encoding_type: str,
            circuit_type: str,
            noise_enabled: bool
    ) -> dict:
        """Create experiment config from parameters."""
        config = json.loads(json.dumps(self.base_config))  # Deep copy

        config["model"]["type"] = model_type
        config["quantum"]["encoding"]["type"] = encoding_type
        config["quantum"]["circuit"]["type"] = circuit_type
        config["quantum"]["noise"]["enabled"] = noise_enabled

        # Create unique experiment name
        noise_str = "noisy" if noise_enabled else "clean"
        config["experiment"]["name"] = f"{model_type}_{encoding_type}_{circuit_type}_{noise_str}"

        return config

    def _get_cache_key(self, encoding_type: str, circuit_type: str, noise_enabled: bool) -> str:
        """Generate cache key for quantum preprocessing."""
        noise_str = "noisy" if noise_enabled else "clean"
        return f"{encoding_type}_{circuit_type}_{noise_str}"

    def _prepare_quantum_data(
            self,
            config: dict,
            dataset_manager: DatasetManager,
            device: torch.device
    ) -> tuple:
        """Prepare quantum-preprocessed data with caching."""
        cache_key = self._get_cache_key(
            config["quantum"]["encoding"]["type"],
            config["quantum"]["circuit"]["type"],
            config["quantum"]["noise"]["enabled"]
        )

        if cache_key in self.quantum_cache:
            return self.quantum_cache[cache_key]

        # Get raw data
        train_images, train_labels, test_images, test_labels = \
            dataset_manager.get_numpy_data(
                train_size=config["dataset"]["train_size"],
                test_size=config["dataset"]["test_size"]
            )

        # Create quanvolutional layer
        quanv_layer = QuanvolutionalLayer(
            n_filters=config["model"]["quanv_filters"],
            kernel_size=config["quantum"]["kernel_size"],
            stride=config["quantum"]["stride"],
            encoding_config=config["quantum"]["encoding"],
            circuit_config=config["quantum"]["circuit"],
            device="default.qubit",
        )

        # Preprocess
        cache_path = Path(config["preprocessing"]["cache_dir"]) / cache_key

        q_train_images, _ = preprocess_for_quantum(
            train_images, train_labels, quanv_layer,
            cache_path=cache_path / "train",
            force_recompute=config["preprocessing"]["force_recompute"]
        )

        q_test_images, _ = preprocess_for_quantum(
            test_images, test_labels, quanv_layer,
            cache_path=cache_path / "test",
            force_recompute=config["preprocessing"]["force_recompute"]
        )

        result = (q_train_images, train_labels, q_test_images, test_labels)
        self.quantum_cache[cache_key] = result

        return result

    def _prepare_dataloaders(
            self,
            images: np.ndarray,
            labels: np.ndarray,
            batch_size: int,
            device: torch.device,
            shuffle: bool = True
    ):
        """Create PyTorch DataLoader from numpy arrays."""
        if images.ndim == 4:
            images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        else:
            images_tensor = torch.from_numpy(images).unsqueeze(1).float()

        labels_tensor = torch.from_numpy(labels).long()
        dataset = torch.utils.data.TensorDataset(images_tensor, labels_tensor)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=(device.type == 'cuda')
        )

    def _run_single_experiment(
            self,
            config: dict,
            dataset_manager: DatasetManager,
            device: torch.device
    ) -> dict:
        """Run a single experiment configuration."""
        model_type = config["model"]["type"]

        if model_type == "qnn":
            # Quantum preprocessing
            q_train, train_labels, q_test, test_labels = \
                self._prepare_quantum_data(config, dataset_manager, device)

            train_loader = self._prepare_dataloaders(
                q_train, train_labels,
                config["training"]["batch_size"],
                device, shuffle=True
            )
            test_loader = self._prepare_dataloaders(
                q_test, test_labels,
                config["training"]["batch_size"],
                device, shuffle=False
            )

            # Create QNN model
            n_qubits = config["quantum"]["n_qubits"]
            n_filters = config["model"]["quanv_filters"]
            quantum_channels = n_qubits * n_filters
            model = get_model("qnn", config["model"], quantum_channels=quantum_channels)

        else:
            # Classical or random nonlinear
            train_loader, test_loader = dataset_manager.get_dataloaders(
                batch_size=config["training"]["batch_size"],
                train_size=config["dataset"]["train_size"],
                test_size=config["dataset"]["test_size"]
            )
            model = get_model(model_type, config["model"])

        # Train
        trainer = Trainer(
            model=model,
            config=config,
            device=device,
            use_wandb=True
        )

        trainer.train(
            train_loader=train_loader,
            val_loader=test_loader,
            num_epochs=config["training"]["epochs"]
        )

        # Evaluate
        test_loss, test_acc = trainer.validate(test_loader)

        return {
            "model": model_type,
            "encoding": config["quantum"]["encoding"]["type"],
            "circuit": config["quantum"]["circuit"]["type"],
            "noise": config["quantum"]["noise"]["enabled"],
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "train_losses": trainer.train_losses,
            "train_accs": trainer.train_accs,
            "val_losses": trainer.val_losses,
            "val_accs": trainer.val_accs,
            "config_name": config["experiment"]["name"]
        }

    def run_sweep(self):
        """Run all sweep experiments."""
        console.print(Panel.fit(
            "[bold cyan]QNN Sweep Experiment[/bold cyan]\n\n"
            f"Models: {self.sweep_config['models']}\n"
            f"Encodings: {self.sweep_config['encodings']}\n"
            f"Circuits: {self.sweep_config['circuits']}\n"
            f"Noise: {self.sweep_config['noise']}",
            border_style="cyan"
        ))

        # Setup
        set_seed(self.base_config["experiment"]["seed"])
        device = get_device(self.base_config["experiment"]["device"])

        dataset_manager = DatasetManager(
            dataset_name=self.base_config["dataset"]["name"],
            data_path=self.base_config["dataset"]["path"],
            normalize=self.base_config["dataset"]["normalize"]
        )

        # Generate all combinations
        combinations = list(itertools.product(
            self.sweep_config["models"],
            self.sweep_config["encodings"],
            self.sweep_config["circuits"],
            self.sweep_config["noise"]
        ))

        # Filter invalid combinations (classical/random don't need quantum params)
        valid_combinations = []
        for model, encoding, circuit, noise in combinations:
            if model in ["classical", "random_nonlinear"]:
                # Only run once per classical model (encoding/circuit don't matter)
                key = (model, "angle", "hardware_efficient", False)
                if key not in valid_combinations:
                    valid_combinations.append(key)
            else:
                valid_combinations.append((model, encoding, circuit, noise))

        total_experiments = len(valid_combinations)
        console.print(f"\n[bold]Running {total_experiments} experiments...[/bold]\n")

        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
        ) as progress:
            task = progress.add_task("Experiments", total=total_experiments)

            for model, encoding, circuit, noise in valid_combinations:
                config = self._create_config(model, encoding, circuit, noise)

                progress.update(
                    task,
                    description=f"[cyan]{config['experiment']['name']}[/cyan]"
                )

                try:
                    result = self._run_single_experiment(
                        config, dataset_manager, device
                    )
                    self.results.append(result)
                    console.print(
                        f"  ✓ {config['experiment']['name']}: "
                        f"{result['test_accuracy']:.2f}%"
                    )
                except Exception as e:
                    console.print(
                        f"  ✗ {config['experiment']['name']}: {str(e)}",
                        style="red"
                    )

                progress.advance(task)

        # Save results
        self._save_results()

        # Generate plots
        self._generate_comparison_plots()

        # Print summary
        self._print_summary()

    def _save_results(self):
        """Save results to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = []
        for r in self.results:
            sr = r.copy()
            sr["train_losses"] = [float(x) for x in r["train_losses"]]
            sr["train_accs"] = [float(x) for x in r["train_accs"]]
            sr["val_losses"] = [float(x) for x in r["val_losses"]]
            sr["val_accs"] = [float(x) for x in r["val_accs"]]
            serializable_results.append(sr)

        results_path = self.results_dir / f"sweep_results_{self.timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        console.print(f"\n[green]Results saved to {results_path}[/green]")

    def _generate_comparison_plots(self):
        """Generate comparative analysis plots."""
        console.print("\n[bold]Generating comparison plots...[/bold]")

        df = pd.DataFrame(self.results)

        plots_dir = self.results_dir / f"plots_{self.timestamp}"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # 1. Vary Model (fix encoding, circuit, noise)
        self._plot_varying_parameter(
            df, "model", ["encoding", "circuit", "noise"],
            "Model Comparison", plots_dir / "vary_model.png"
        )

        # 2. Vary Encoding (fix model=qnn, circuit, noise)
        qnn_df = df[df["model"] == "qnn"]
        if not qnn_df.empty:
            self._plot_varying_parameter(
                qnn_df, "encoding", ["circuit", "noise"],
                "Encoding Comparison (QNN)", plots_dir / "vary_encoding.png"
            )

        # 3. Vary Circuit (fix model=qnn, encoding, noise)
        if not qnn_df.empty:
            self._plot_varying_parameter(
                qnn_df, "circuit", ["encoding", "noise"],
                "Circuit Comparison (QNN)", plots_dir / "vary_circuit.png"
            )

        # 4. Vary Noise (fix model=qnn, encoding, circuit)
        if not qnn_df.empty:
            self._plot_varying_parameter(
                qnn_df, "noise", ["encoding", "circuit"],
                "Noise Impact (QNN)", plots_dir / "vary_noise.png"
            )

        # 5. Comprehensive heatmap for QNN
        if not qnn_df.empty:
            self._plot_heatmaps(qnn_df, plots_dir)

        # 6. Training curves comparison
        self._plot_training_curves_grid(df, plots_dir / "training_curves_grid.png")

        console.print(f"[green]Plots saved to {plots_dir}[/green]")

    def _plot_varying_parameter(
            self,
            df: pd.DataFrame,
            vary_param: str,
            fixed_params: list,
            title: str,
            save_path: Path
    ):
        """Plot accuracy varying one parameter, grouping by fixed parameters."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create grouping key from fixed parameters
        if fixed_params:
            df = df.copy()
            df["group"] = df[fixed_params].astype(str).agg("_".join, axis=1)
            groups = df["group"].unique()

            x = np.arange(len(df[vary_param].unique()))
            width = 0.8 / len(groups)

            colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))

            for i, group in enumerate(groups):
                group_df = df[df["group"] == group]
                values = []
                labels = []

                for param_val in df[vary_param].unique():
                    subset = group_df[group_df[vary_param] == param_val]
                    if not subset.empty:
                        values.append(subset["test_accuracy"].mean())
                        labels.append(str(param_val))
                    else:
                        values.append(0)
                        labels.append(str(param_val))

                bars = ax.bar(
                    x + i * width,
                    values,
                    width,
                    label=group,
                    color=colors[i],
                    alpha=0.8
                )

                # Add value labels
                for bar, val in zip(bars, values):
                    if val > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.5,
                            f'{val:.1f}',
                            ha='center',
                            va='bottom',
                            fontsize=8
                        )

            ax.set_xticks(x + width * (len(groups) - 1) / 2)
            ax.set_xticklabels(df[vary_param].unique())
            ax.legend(title="_".join(fixed_params), bbox_to_anchor=(1.05, 1), loc='upper left')

        else:
            # Simple bar plot
            grouped = df.groupby(vary_param)["test_accuracy"].mean()
            bars = ax.bar(range(len(grouped)), grouped.values, color='steelblue', alpha=0.8)
            ax.set_xticks(range(len(grouped)))
            ax.set_xticklabels(grouped.index)

            for bar, val in zip(bars, grouped.values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f'{val:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=10
                )

        ax.set_xlabel(vary_param.capitalize(), fontsize=12)
        ax.set_ylabel("Test Accuracy (%)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim([0, 105])
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def _plot_heatmaps(self, df: pd.DataFrame, plots_dir: Path):
        """Generate heatmaps for parameter combinations."""
        # Encoding vs Circuit (no noise)
        clean_df = df[df["noise"] == False]
        if not clean_df.empty:
            pivot = clean_df.pivot_table(
                values="test_accuracy",
                index="encoding",
                columns="circuit",
                aggfunc="mean"
            )

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                pivot,
                annot=True,
                fmt='.1f',
                cmap='YlOrRd',
                ax=ax,
                cbar_kws={'label': 'Accuracy (%)'}
            )
            ax.set_title("QNN: Encoding vs Circuit (Clean)", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(plots_dir / "heatmap_encoding_circuit_clean.png", dpi=300, facecolor='white')
            plt.close()

        # Encoding vs Circuit (noisy)
        noisy_df = df[df["noise"] == True]
        if not noisy_df.empty:
            pivot = noisy_df.pivot_table(
                values="test_accuracy",
                index="encoding",
                columns="circuit",
                aggfunc="mean"
            )

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                pivot,
                annot=True,
                fmt='.1f',
                cmap='YlOrRd',
                ax=ax,
                cbar_kws={'label': 'Accuracy (%)'}
            )
            ax.set_title("QNN: Encoding vs Circuit (Noisy)", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(plots_dir / "heatmap_encoding_circuit_noisy.png", dpi=300, facecolor='white')
            plt.close()

        # Noise impact heatmap
        if not clean_df.empty and not noisy_df.empty:
            clean_pivot = clean_df.pivot_table(
                values="test_accuracy",
                index="encoding",
                columns="circuit",
                aggfunc="mean"
            )
            noisy_pivot = noisy_df.pivot_table(
                values="test_accuracy",
                index="encoding",
                columns="circuit",
                aggfunc="mean"
            )

            diff_pivot = clean_pivot - noisy_pivot

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                diff_pivot,
                annot=True,
                fmt='.1f',
                cmap='RdYlGn',
                center=0,
                ax=ax,
                cbar_kws={'label': 'Accuracy Drop (%)'}
            )
            ax.set_title("Noise Impact: Clean - Noisy Accuracy", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(plots_dir / "heatmap_noise_impact.png", dpi=300, facecolor='white')
            plt.close()

    def _plot_training_curves_grid(self, df: pd.DataFrame, save_path: Path):
        """Plot training curves for all experiments in a grid."""
        n_experiments = len(df)
        n_cols = 4
        n_rows = (n_experiments + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

        for idx, (_, row) in enumerate(df.iterrows()):
            ax = axes[idx]
            epochs = range(1, len(row["train_accs"]) + 1)

            ax.plot(epochs, row["train_accs"], label="Train", color='blue', alpha=0.7)
            ax.plot(epochs, row["val_accs"], label="Val", color='red', linewidth=2)

            ax.set_title(row["config_name"], fontsize=9)
            ax.set_xlabel("Epoch", fontsize=8)
            ax.set_ylabel("Accuracy (%)", fontsize=8)
            ax.set_ylim([0, 105])
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_experiments, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()

    def _print_summary(self):
        """Print experiment summary table."""
        console.print("\n" + "=" * 80)
        console.print("[bold]SWEEP EXPERIMENT SUMMARY[/bold]")
        console.print("=" * 80 + "\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Encoding", style="yellow")
        table.add_column("Circuit", style="green")
        table.add_column("Noise", style="red")
        table.add_column("Accuracy", justify="right", style="bold")

        # Sort by accuracy
        sorted_results = sorted(self.results, key=lambda x: x["test_accuracy"], reverse=True)

        for r in sorted_results:
            noise_str = "Yes" if r["noise"] else "No"
            table.add_row(
                r["model"],
                r["encoding"],
                r["circuit"],
                noise_str,
                f"{r['test_accuracy']:.2f}%"
            )

        console.print(table)

        # Best configurations
        if sorted_results:
            best = sorted_results[0]
            console.print(Panel.fit(
                f"[bold green]Best Configuration:[/bold green]\n\n"
                f"Model: {best['model']}\n"
                f"Encoding: {best['encoding']}\n"
                f"Circuit: {best['circuit']}\n"
                f"Noise: {'Yes' if best['noise'] else 'No'}\n"
                f"Accuracy: {best['test_accuracy']:.2f}%",
                border_style="green"
            ))


def main():
    """Run sweep experiments."""
    sweep = SweepExperiment(
        base_config=BASE_CONFIG,
        sweep_config=SWEEP_CONFIG,
        results_dir="./results/sweep"
    )

    sweep.run_sweep()


if __name__ == "__main__":
    main()
