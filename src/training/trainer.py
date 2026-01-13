"""
Training and evaluation logic for QNN experiments.
Includes WandB integration and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import wandb
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
import numpy as np

console = Console()


class Trainer:
    """Trainer class for QNN experiments."""

    def __init__(
            self,
            model: nn.Module,
            config: Dict[str, Any],
            device: torch.device,
            use_wandb: bool = True,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.use_wandb = use_wandb

        # Setup optimizer
        self.optimizer = self._get_optimizer()

        # Setup scheduler if specified
        self.scheduler = self._get_scheduler()

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics storage
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []

        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_path = None

        # Checkpoint directory
        self.checkpoint_dir = Path(config["training"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Build checkpoint filename prefix
        self.checkpoint_prefix = self._build_checkpoint_prefix()

    def _build_checkpoint_prefix(self) -> str:
        """Build checkpoint filename prefix from config."""
        model_type = self.config.get("model", {}).get("type", "unknown")
        encoding_type = self.config.get("quantum", {}).get("encoding", {}).get("type", "unknown")
        circuit_type = self.config.get("quantum", {}).get("circuit", {}).get("type", "unknown")
        noise_enabled = self.config.get("quantum", {}).get("noise", {}).get("enabled", False)
        noise_str = "noisy" if noise_enabled else "clean"

        return f"{model_type}_{encoding_type}_{circuit_type}_{noise_str}"

    def _get_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        opt_name = self.config["training"]["optimizer"].lower()
        lr = self.config["training"]["learning_rate"]

        if opt_name == "adam":
            return optim.Adam(self.model.parameters(), lr=lr)
        elif opt_name == "sgd":
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        elif opt_name == "adamw":
            return optim.AdamW(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def _get_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler if specified."""
        sched_name = self.config["training"].get("scheduler")

        if sched_name is None:
            return None
        elif sched_name == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.1
            )
        elif sched_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config["training"]["epochs"]
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched_name}")

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Training...", total=len(train_loader)
            )

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Metrics
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

                progress.update(task, advance=1)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def train(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            num_epochs: int,
    ):
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
        """
        console.print("\n[bold green]Starting Training[/bold green]\n")

        for epoch in range(1, num_epochs + 1):
            console.print(f"\n[bold]Epoch {epoch}/{num_epochs}[/bold]")

            # Training
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validation
            val_loss, val_acc = self.validate(val_loader)

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # Display metrics
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Split", style="cyan")
            table.add_column("Loss", justify="right")
            table.add_column("Accuracy", justify="right")

            table.add_row("Train", f"{train_loss:.4f}", f"{train_acc:.2f}%")
            table.add_row("Val", f"{val_loss:.4f}", f"{val_acc:.2f}%")

            console.print(table)

            # WandB logging
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": self.optimizer.param_groups[0]["lr"],
                })

            # Save checkpoint
            if epoch % self.config["training"]["save_frequency"] == 0:
                self.save_checkpoint(epoch, val_acc)

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_best_model(epoch, val_acc)

        console.print(
            f"\n[bold green]Training Complete![/bold green] "
            f"Best Val Acc: {self.best_val_acc:.2f}%"
        )

    def save_checkpoint(self, epoch: int, val_acc: float):
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{self.checkpoint_prefix}_epoch_{epoch}.pt"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
        }, checkpoint_path)

        console.print(f"[dim]Saved checkpoint to {checkpoint_path}[/dim]")

    def save_best_model(self, epoch: int, val_acc: float):
        """Save best model."""
        model_path = self.checkpoint_dir / f"{self.checkpoint_prefix}_best.pt"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_acc': val_acc,
        }, model_path)

        self.best_model_path = model_path
        console.print(f"[bold green]✓[/bold green] New best model saved! Val Acc: {val_acc:.2f}%")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.train_accs = checkpoint.get('train_accs', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accs = checkpoint.get('val_accs', [])

        console.print(f"[green]Loaded checkpoint from {checkpoint_path}[/green]")

        return checkpoint['epoch']

    def get_predictions(
            self,
            data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions for entire dataset.

        Returns:
            Tuple of (predictions, true_labels)
        """
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1).cpu().numpy()

                all_preds.append(pred)
                all_labels.append(target.numpy())

        return np.concatenate(all_preds), np.concatenate(all_labels)
