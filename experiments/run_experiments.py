"""
Main experiment runner for QNN research.
Orchestrates data loading, quantum preprocessing, training, and evaluation.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import yaml
import argparse
from rich.console import Console
from rich.panel import Panel
import wandb

from src.data.dataset import DatasetManager, preprocess_for_quantum
from src.quantum.quanvolution import QuanvolutionalLayer
from src.models.qnn import get_model
from src.training.trainer import Trainer
from src.visualization.plots import Visualizer
from src.utils.helpers import set_seed, get_device, print_config

console = Console()


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_experiment(config: dict) -> dict:
    """
    Setup experiment environment.

    Returns:
        Dict with device and other setup info
    """
    # Set random seed
    set_seed(config["experiment"]["seed"])

    # Get device
    device = get_device(config["experiment"]["device"])

    console.print(Panel.fit(
        f"[bold green]Experiment Setup[/bold green]\n"
        f"Name: {config['experiment']['name']}\n"
        f"Device: {device}\n"
        f"Dataset: {config['dataset']['name']}\n"
        f"Model: {config['model']['type']}",
        border_style="green"
    ))

    return {"device": device}


def create_quanvolutional_layer(config: dict) -> QuanvolutionalLayer:
    """Create and configure quanvolutional layer."""
    console.print("\n[bold cyan]Creating Quanvolutional Layer...[/bold cyan]")

    quanv_layer = QuanvolutionalLayer(
        n_filters=config["model"]["quanv_filters"],
        kernel_size=config["quantum"]["kernel_size"],
        stride=config["quantum"]["stride"],
        encoding_config=config["quantum"]["encoding"],
        circuit_config=config["quantum"]["circuit"],
        device="default.qubit",
    )

    console.print(
        f"✓ Created {config['model']['quanv_filters']} quantum filters "
        f"with {config['quantum']['n_qubits']} qubits each"
    )

    return quanv_layer


def run_quantum_preprocessing(
        config: dict,
        dataset_manager: DatasetManager
) -> tuple:
    """
    Run quantum preprocessing on dataset.

    Returns:
        Tuple of (q_train_images, train_labels, q_test_images, test_labels)
    """
    console.print("\n[bold cyan]Quantum Preprocessing...[/bold cyan]")

    # Load raw data
    train_images, train_labels, test_images, test_labels = \
        dataset_manager.get_numpy_data(
            train_size=config["dataset"]["train_size"],
            test_size=config["dataset"]["test_size"]
        )

    # Create quanvolutional layer
    quanv_layer = create_quanvolutional_layer(config)

    # Preprocess with caching
    cache_path = Path(config["preprocessing"]["cache_dir"])
    cache_path.mkdir(parents=True, exist_ok=True)

    q_train_images, train_labels = preprocess_for_quantum(
        train_images,
        train_labels,
        quanv_layer,
        cache_path=cache_path / "train",
        force_recompute=config["preprocessing"]["force_recompute"]
    )

    q_test_images, test_labels = preprocess_for_quantum(
        test_images,
        test_labels,
        quanv_layer,
        cache_path=cache_path / "test",
        force_recompute=config["preprocessing"]["force_recompute"]
    )

    console.print(
        f"✓ Preprocessed {len(q_train_images)} train and "
        f"{len(q_test_images)} test images"
    )
    console.print(
        f"  Output shape: {q_train_images.shape}"
    )

    return q_train_images, train_labels, q_test_images, test_labels


def prepare_dataloaders(
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        device: torch.device,
        shuffle: bool = True
) -> torch.utils.data.DataLoader:
    """Create PyTorch DataLoader from numpy arrays."""
    # Convert to torch tensors
    # images: [N, H, W, C] -> [N, C, H, W]
    images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float()
    labels_tensor = torch.from_numpy(labels).long()

    dataset = torch.utils.data.TensorDataset(images_tensor, labels_tensor)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=(device.type == 'cuda')
    )

    return loader


def train_model(
        model_type: str,
        config: dict,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        use_wandb: bool = True
) -> Trainer:
    """Train a model and return the trainer."""
    console.print(f"\n[bold green]Training {model_type.upper()} Model[/bold green]")

    # Create model
    if model_type == "qnn":
        n_qubits = config["quantum"]["n_qubits"]
        n_filters = config["model"]["quanv_filters"]
        quantum_channels = n_qubits * n_filters
        model = get_model(
            "qnn",
            config["model"],
            quantum_channels=quantum_channels
        )
    else:
        model = get_model(model_type, config["model"])

    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        use_wandb=use_wandb
    )

    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config["training"]["epochs"]
    )

    return trainer


def main(config_path: str = "config/config.yaml"):
    """Main experiment function."""
    # Load configuration
    config = load_config(config_path)

    # Setup experiment
    setup_info = setup_experiment(config)
    device = setup_info["device"]

    # Initialize WandB
    if config["wandb"]["enabled"]:
        wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"]["entity"],
            config=config,
            name=config["experiment"]["name"],
            tags=config["wandb"]["tags"]
        )

    # Create dataset manager
    dataset_manager = DatasetManager(
        dataset_name=config["dataset"]["name"],
        data_path=config["dataset"]["path"],
        normalize=config["dataset"]["normalize"]
    )

    # Initialize visualizer
    viz = Visualizer(
        save_dir=config["visualization"]["save_dir"],
        use_wandb=config["wandb"]["enabled"]
    )

    # Run based on model type
    if config["model"]["type"] == "qnn":
        # Quantum preprocessing
        q_train_images, train_labels, q_test_images, test_labels = \
            run_quantum_preprocessing(config, dataset_manager)

        # Visualize feature maps
        if config["visualization"]["plot_feature_maps"]:
            # Get some original images for comparison
            raw_train, _, _, _ = dataset_manager.get_numpy_data(
                train_size=config["visualization"]["n_samples"],
                test_size=1
            )
            if raw_train.ndim == 4:
                raw_train = raw_train[:, 0, :, :]  # [N, C, H, W] -> [N, H, W]

            viz.plot_feature_maps(
                original_images=raw_train,
                quantum_features=q_train_images[:config["visualization"]["n_samples"]],
                n_samples=config["visualization"]["n_samples"]
            )

        # Create dataloaders
        train_loader = prepare_dataloaders(
            q_train_images, train_labels,
            config["training"]["batch_size"],
            device,
            shuffle=True
        )

        test_loader = prepare_dataloaders(
            q_test_images, test_labels,
            config["training"]["batch_size"],
            device,
            shuffle=False
        )

        # Train QNN
        qnn_trainer = train_model(
            "qnn",
            config,
            train_loader,
            test_loader,
            device,
            use_wandb=config["wandb"]["enabled"]
        )

        # Evaluate
        final_loss, final_acc = qnn_trainer.validate(test_loader)
        console.print(
            f"\n[bold green]Final QNN Test Accuracy: {final_acc:.2f}%[/bold green]"
        )

        # Plot results
        if config["visualization"]["plot_metrics"]:
            viz.plot_training_curves({
                'qnn': {
                    'train_losses': qnn_trainer.train_losses,
                    'train_accs': qnn_trainer.train_accs,
                    'val_losses': qnn_trainer.val_losses,
                    'val_accs': qnn_trainer.val_accs,
                }
            })

        # Confusion matrix
        if config["visualization"]["plot_confusion_matrix"]:
            preds, labels = qnn_trainer.get_predictions(test_loader)
            viz.plot_confusion_matrix(preds, labels)

    elif config["model"]["type"] == "classical":
        # Load data normally (no quantum preprocessing)
        train_loader, test_loader = dataset_manager.get_dataloaders(
            batch_size=config["training"]["batch_size"],
            train_size=config["dataset"]["train_size"],
            test_size=config["dataset"]["test_size"]
        )

        # Train classical CNN
        classical_trainer = train_model(
            "classical",
            config,
            train_loader,
            test_loader,
            device,
            use_wandb=config["wandb"]["enabled"]
        )

        final_loss, final_acc = classical_trainer.validate(test_loader)
        console.print(
            f"\n[bold green]Final Classical CNN Test Accuracy: {final_acc:.2f}%[/bold green]"
        )

        if config["visualization"]["plot_metrics"]:
            viz.plot_training_curves({
                'classical': {
                    'train_losses': classical_trainer.train_losses,
                    'train_accs': classical_trainer.train_accs,
                    'val_losses': classical_trainer.val_losses,
                    'val_accs': classical_trainer.val_accs,
                }
            })

    # Close WandB
    if config["wandb"]["enabled"]:
        wandb.finish()

    console.print("\n[bold green]Experiment Complete! ✓[/bold green]\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QNN experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file"
    )

    args = parser.parse_args()
    main(args.config)