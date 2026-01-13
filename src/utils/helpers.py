"""
Utility functions for QNN experiments.
"""

import torch
import numpy as np
import random
from rich.console import Console
from rich.table import Table
from typing import Dict, Any
import yaml

console = Console()


def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CUDA deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto") -> torch.device:
    """
    Get torch device based on availability.

    Args:
        device_str: Device specification ("auto", "cuda", "cpu")

    Returns:
        torch.device
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if device.type == "cuda":
        console.print(
            f"[green]✓[/green] Using GPU: {torch.cuda.get_device_name(0)}"
        )
        console.print(
            f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        console.print("[yellow]⚠[/yellow] Using CPU (GPU not available)")

    return device


def print_config(config: Dict[str, Any]):
    """
    Pretty print configuration.

    Args:
        config: Configuration dictionary
    """
    table = Table(title="Experiment Configuration", show_header=True)
    table.add_column("Section", style="cyan", no_wrap=True)
    table.add_column("Parameter", style="magenta")
    table.add_column("Value", style="green")

    def add_rows(section_name: str, section_dict: dict, indent: int = 0):
        """Recursively add configuration rows."""
        for key, value in section_dict.items():
            if isinstance(value, dict):
                table.add_row(
                    section_name if indent == 0 else "",
                    "  " * indent + key,
                    ""
                )
                add_rows("", value, indent + 1)
            else:
                table.add_row(
                    section_name if indent == 0 else "",
                    "  " * indent + key,
                    str(value)
                )

    for section, params in config.items():
        if isinstance(params, dict):
            add_rows(section, params)
        else:
            table.add_row(section, "", str(params))

    console.print(table)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: torch.nn.Module, model_name: str = "Model"):
    """
    Print model information.

    Args:
        model: PyTorch model
        model_name: Name of the model
    """
    n_params = count_parameters(model)

    console.print(f"\n[bold]{model_name} Information:[/bold]")
    console.print(f"  Total parameters: {n_params:,}")
    console.print(f"  Model size: ~{n_params * 4 / 1e6:.2f} MB (float32)")


def calculate_output_size(
        input_size: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0
) -> int:
    """
    Calculate output size after convolution/pooling.

    Args:
        input_size: Input dimension
        kernel_size: Kernel size
        stride: Stride
        padding: Padding

    Returns:
        Output dimension
    """
    return (input_size - kernel_size + 2 * padding) // stride + 1


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_results(results: Dict[str, Any], filepath: str):
    """
    Save experiment results to file.

    Args:
        results: Dictionary of results
        filepath: Path to save file
    """
    import json
    from pathlib import Path

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    console.print(f"[green]✓[/green] Results saved to {filepath}")


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load experiment results from file.

    Args:
        filepath: Path to results file

    Returns:
        Dictionary of results
    """
    import json

    with open(filepath, 'r') as f:
        results = json.load(f)

    return results


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config