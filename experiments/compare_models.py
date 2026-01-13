"""
Compare QNN, Classical CNN, and Random Nonlinear models.
Reproduces the comparison from Henderson et al. (2019).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import wandb

from src.data.dataset import DatasetManager, preprocess_for_quantum
from src.quantum.quanvolution import QuanvolutionalLayer
from src.models.qnn import get_model
from src.training.trainer import Trainer
from src.visualization.plots import Visualizer
from src.utils.helpers import set_seed, get_device

console = Console()


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_dataloaders(
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
        num_workers=4,
        pin_memory=(device.type == 'cuda')
    )


def main():
    """Run comparison experiment."""
    console.print(Panel.fit(
        "[bold cyan]Quanvolutional Neural Networks[/bold cyan]\n"
        "[dim]Model Comparison Experiment[/dim]\n\n"
        "Comparing three models:\n"
        "  1. QNN (Quanvolutional + Classical)\n"
        "  2. Classical CNN (Baseline)\n"
        "  3. Random Nonlinear CNN",
        border_style="cyan"
    ))

    # Load config
    config = load_config("../config/config.yaml")
    set_seed(config["experiment"]["seed"])
    device = get_device(config["experiment"]["device"])

    # Initialize WandB
    if config["wandb"]["enabled"]:
        wandb.init(
            project=config["wandb"]["project"],
            name="qnn_comparison",
            config=config,
            tags=["comparison", "all_models"]
        )

    # Dataset manager
    dataset_manager = DatasetManager(
        dataset_name=config["dataset"]["name"],
        data_path=config["dataset"]["path"],
        normalize=config["dataset"]["normalize"]
    )

    # Visualizer
    viz = Visualizer(
        save_dir=config["visualization"]["save_dir"],
        use_wandb=config["wandb"]["enabled"],
        config=config
    )

    results = {}
    all_metrics = {}

    # ===================================================================
    # 1. CLASSICAL CNN (Baseline)
    # ===================================================================
    console.print("\n[bold green]═══ Training Classical CNN ═══[/bold green]\n")

    train_loader, test_loader = dataset_manager.get_dataloaders(
        batch_size=config["training"]["batch_size"],
        train_size=config["dataset"]["train_size"],
        test_size=config["dataset"]["test_size"]
    )

    classical_model = get_model("classical", config["model"])
    classical_trainer = Trainer(
        model=classical_model,
        config=config,
        device=device,
        use_wandb=False  # Will log manually
    )

    classical_trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=config["training"]["epochs"]
    )

    classical_loss, classical_acc = classical_trainer.validate(test_loader)
    results['Classical CNN'] = classical_acc
    all_metrics['classical'] = {
        'train_losses': classical_trainer.train_losses,
        'train_accs': classical_trainer.train_accs,
        'val_losses': classical_trainer.val_losses,
        'val_accs': classical_trainer.val_accs,
    }

    console.print(f"\n[green]✓ Classical CNN: {classical_acc:.2f}%[/green]")

    # ===================================================================
    # 2. QUANTUM CNN (QNN)
    # ===================================================================
    console.print("\n[bold cyan]═══ Training Quantum CNN ═══[/bold cyan]\n")

    # Get raw data for quantum preprocessing
    train_images, train_labels, test_images, test_labels = \
        dataset_manager.get_numpy_data(
            train_size=config["dataset"]["train_size"],
            test_size=config["dataset"]["test_size"]
        )

    # Create quanvolutional layer
    console.print("[dim]Creating quanvolutional layer...[/dim]")
    quanv_layer = QuanvolutionalLayer(
        n_filters=config["model"]["quanv_filters"],
        kernel_size=config["quantum"]["kernel_size"],
        stride=config["quantum"]["stride"],
        encoding_config=config["quantum"]["encoding"],
        circuit_config=config["quantum"]["circuit"],
        device="default.qubit",
    )

    # Preprocess
    cache_path = Path(config["preprocessing"]["cache_dir"])
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

    # Visualize quantum features
    if config["visualization"]["plot_feature_maps"]:
        raw_sample = train_images[:8]
        if raw_sample.ndim == 4:
            raw_sample = raw_sample[:, 0, :, :]

        viz.plot_feature_maps(
            original_images=raw_sample,
            quantum_features=q_train_images[:8],
            n_samples=8,
            save_name="quantum_vs_original.png"
        )

    # Create dataloaders
    q_train_loader = prepare_dataloaders(
        q_train_images, train_labels,
        config["training"]["batch_size"],
        device, shuffle=True
    )

    q_test_loader = prepare_dataloaders(
        q_test_images, test_labels,
        config["training"]["batch_size"],
        device, shuffle=False
    )

    # Train QNN
    n_qubits = config["quantum"]["n_qubits"]
    n_filters = config["model"]["quanv_filters"]
    quantum_channels = n_qubits * n_filters

    qnn_model = get_model("qnn", config["model"], quantum_channels=quantum_channels)
    qnn_trainer = Trainer(
        model=qnn_model,
        config=config,
        device=device,
        use_wandb=False
    )

    qnn_trainer.train(
        train_loader=q_train_loader,
        val_loader=q_test_loader,
        num_epochs=config["training"]["epochs"]
    )

    qnn_loss, qnn_acc = qnn_trainer.validate(q_test_loader)
    results['QNN'] = qnn_acc
    all_metrics['qnn'] = {
        'train_losses': qnn_trainer.train_losses,
        'train_accs': qnn_trainer.train_accs,
        'val_losses': qnn_trainer.val_losses,
        'val_accs': qnn_trainer.val_accs,
    }

    console.print(f"\n[cyan]✓ QNN: {qnn_acc:.2f}%[/cyan]")

    # ===================================================================
    # 3. RANDOM NONLINEAR CNN
    # ===================================================================
    console.print("\n[bold yellow]═══ Training Random Nonlinear CNN ═══[/bold yellow]\n")

    random_model = get_model("random_nonlinear", config["model"])
    random_trainer = Trainer(
        model=random_model,
        config=config,
        device=device,
        use_wandb=False
    )

    random_trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=config["training"]["epochs"]
    )

    random_loss, random_acc = random_trainer.validate(test_loader)
    results['Random Nonlinear'] = random_acc
    all_metrics['random'] = {
        'train_losses': random_trainer.train_losses,
        'train_accs': random_trainer.train_accs,
        'val_losses': random_trainer.val_losses,
        'val_accs': random_trainer.val_accs,
    }

    console.print(f"\n[yellow]✓ Random Nonlinear: {random_acc:.2f}%[/yellow]")

    # ===================================================================
    # RESULTS SUMMARY
    # ===================================================================
    console.print("\n" + "=" * 60)
    console.print("[bold]FINAL RESULTS[/bold]")
    console.print("=" * 60 + "\n")

    # Create results table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Test Accuracy", justify="right", style="green")
    table.add_column("Relative to Classical", justify="right")

    classical_baseline = results['Classical CNN']

    for model_name, accuracy in results.items():
        diff = accuracy - classical_baseline
        diff_str = f"+{diff:.2f}%" if diff > 0 else f"{diff:.2f}%"
        if abs(diff) < 0.5:
            diff_str = "~" + diff_str

        table.add_row(
            model_name,
            f"{accuracy:.2f}%",
            diff_str
        )

    console.print(table)

    # Generate visualizations
    console.print("\n[dim]Generating visualizations...[/dim]")

    # Training curves
    viz.plot_training_curves(
        all_metrics,
        save_name="all_models_comparison.png"
    )

    # Bar chart
    viz.plot_comparison_bar(
        results,
        save_name="accuracy_comparison.png"
    )

    # Confusion matrices
    for model_name, trainer in [
        ('QNN', qnn_trainer),
        ('Classical', classical_trainer),
        ('Random', random_trainer)
    ]:
        if model_name == 'QNN':
            loader = q_test_loader
        else:
            loader = test_loader

        preds, labels = trainer.get_predictions(loader)
        viz.plot_confusion_matrix(
            preds, labels,
            save_name=f"confusion_{model_name.lower()}.png"
        )

    # Log to WandB
    if config["wandb"]["enabled"]:
        wandb.log({
            "classical_acc": classical_acc,
            "qnn_acc": qnn_acc,
            "random_acc": random_acc,
            "qnn_improvement": qnn_acc - classical_acc,
        })
        wandb.finish()

    console.print(
        f"\n[bold green]✓ Comparison Complete![/bold green] "
        f"Results saved to {config['visualization']['save_dir']}\n"
    )

    # Key findings
    console.print(Panel.fit(
        "[bold]Key Findings:[/bold]\n\n"
        f"• QNN vs Classical: {qnn_acc - classical_acc:+.2f}%\n"
        f"• Random vs Classical: {random_acc - classical_acc:+.2f}%\n"
        f"• QNN vs Random: {qnn_acc - random_acc:+.2f}%\n\n"
        "[dim]See Henderson et al. (2019) for interpretation[/dim]",
        border_style="green"
    ))


if __name__ == "__main__":
    main()