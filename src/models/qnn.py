"""
Neural network models for QNN experiments.
Includes QNN, classical CNN, and random non-linear baselines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class ClassicalCNN(nn.Module):
    """
    Classical CNN baseline as described in Henderson et al. 2019.

    Architecture:
    CONV1 - POOL1 - CONV2 - POOL2 - FC1 - FC2
    """

    def __init__(
            self,
            input_channels: int = 1,
            conv1_filters: int = 50,
            conv1_kernel: int = 5,
            conv2_filters: int = 64,
            conv2_kernel: int = 5,
            fc1_units: int = 1024,
            dropout: float = 0.4,
            num_classes: int = 10,
    ):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            input_channels, conv1_filters,
            kernel_size=conv1_kernel,
            padding=conv1_kernel // 2
        )
        self.conv2 = nn.Conv2d(
            conv1_filters, conv2_filters,
            kernel_size=conv2_kernel,
            padding=conv2_kernel // 2
        )

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate flattened size (depends on input size)
        # For 28x28 input: 28->14->7, so 7*7*conv2_filters
        self.fc1_input_size = 7 * 7 * conv2_filters

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc1_input_size, fc1_units)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc1_units, num_classes)

    def forward(self, x):
        # Conv1 + ReLU + Pool
        x = self.pool(F.relu(self.conv1(x)))

        # Conv2 + ReLU + Pool
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten
        x = x.view(-1, self.fc1_input_size)

        # FC1 + ReLU + Dropout
        x = self.dropout(F.relu(self.fc1(x)))

        # FC2 (output)
        x = self.fc2(x)

        return x


class QuantumCNN(nn.Module):
    """
    Quantum CNN (QNN) with classical layers after quantum preprocessing.

    Architecture:
    [QUANTUM PREPROCESSING] -> CONV1 - POOL1 - CONV2 - POOL2 - FC1 - FC2

    Note: Quantum preprocessing is done offline, so this model receives
    already-preprocessed quantum features.
    """

    def __init__(
            self,
            quantum_channels: int,  # n_filters * n_qubits from quanv layer
            conv1_filters: int = 50,
            conv1_kernel: int = 5,
            conv2_filters: int = 64,
            conv2_kernel: int = 5,
            fc1_units: int = 1024,
            dropout: float = 0.4,
            num_classes: int = 10,
    ):
        super().__init__()

        # Input already has quantum_channels from quanvolution
        self.conv1 = nn.Conv2d(
            quantum_channels, conv1_filters,
            kernel_size=conv1_kernel,
            padding=conv1_kernel // 2
        )
        self.conv2 = nn.Conv2d(
            conv1_filters, conv2_filters,
            kernel_size=conv2_kernel,
            padding=conv2_kernel // 2
        )

        self.pool = nn.MaxPool2d(2, 2)

        # For 14x14 input (from 2x2 quanv with stride 2): 14->7->3
        self.fc1_input_size = 3 * 3 * conv2_filters

        self.fc1 = nn.Linear(self.fc1_input_size, fc1_units)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc1_units, num_classes)

    def forward(self, x):
        # Input is already quantum-preprocessed
        # Shape: [batch, quantum_channels, H', W']

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


class SimpleClassifier(nn.Module):
    """
    Simple fully-connected classifier for quantum features.
    Used in the PennyLane tutorial.
    """

    def __init__(
            self,
            input_size: int,
            num_classes: int = 10,
    ):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x


class RandomNonlinearCNN(nn.Module):
    """
    CNN with random non-linear transformation (classical baseline).
    Used to compare against quantum transformations.
    """

    def __init__(
            self,
            input_channels: int = 1,
            random_filters: int = 25,
            conv1_filters: int = 50,
            conv1_kernel: int = 5,
            conv2_filters: int = 64,
            conv2_kernel: int = 5,
            fc1_units: int = 1024,
            dropout: float = 0.4,
            num_classes: int = 10,
            seed: int = 42,
    ):
        super().__init__()

        # Random non-linear transformation (fixed, not trained)
        torch.manual_seed(seed)
        self.random_conv = nn.Conv2d(
            input_channels, random_filters,
            kernel_size=2, stride=2
        )
        # Freeze random layer
        for param in self.random_conv.parameters():
            param.requires_grad = False

        # Rest of the network (trainable)
        self.conv1 = nn.Conv2d(
            random_filters, conv1_filters,
            kernel_size=conv1_kernel,
            padding=conv1_kernel // 2
        )
        self.conv2 = nn.Conv2d(
            conv1_filters, conv2_filters,
            kernel_size=conv2_kernel,
            padding=conv2_kernel // 2
        )

        self.pool = nn.MaxPool2d(2, 2)

        # Similar to QNN
        self.fc1_input_size = 3 * 3 * conv2_filters

        self.fc1 = nn.Linear(self.fc1_input_size, fc1_units)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc1_units, num_classes)

    def forward(self, x):
        # Random transformation (fixed)
        x = torch.tanh(self.random_conv(x))  # Non-linear activation

        # Trainable layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


def get_model(model_type: str, config: Dict[str, Any], **kwargs) -> nn.Module:
    """
    Factory function to get model by type.

    Args:
        model_type: Type of model (qnn, classical, random_nonlinear)
        config: Model configuration
        **kwargs: Additional parameters

    Returns:
        Model instance
    """
    if model_type == "classical":
        return ClassicalCNN(
            input_channels=kwargs.get("input_channels", 1),
            conv1_filters=config.get("conv1_filters", 50),
            conv1_kernel=config.get("conv1_kernel", 5),
            conv2_filters=config.get("conv2_filters", 64),
            conv2_kernel=config.get("conv2_kernel", 5),
            fc1_units=config.get("fc1_units", 1024),
            dropout=config.get("dropout", 0.4),
            num_classes=config.get("num_classes", 10),
        )

    elif model_type == "qnn":
        quantum_channels = kwargs.get("quantum_channels")
        if quantum_channels is None:
            # Calculate from quanv config
            n_qubits = config.get("quantum", {}).get("n_qubits", 4)
            n_filters = config.get("quanv_filters", 25)
            quantum_channels = n_qubits * n_filters

        return QuantumCNN(
            quantum_channels=quantum_channels,
            conv1_filters=config.get("conv1_filters", 50),
            conv1_kernel=config.get("conv1_kernel", 5),
            conv2_filters=config.get("conv2_filters", 64),
            conv2_kernel=config.get("conv2_kernel", 5),
            fc1_units=config.get("fc1_units", 1024),
            dropout=config.get("dropout", 0.4),
            num_classes=config.get("num_classes", 10),
        )

    elif model_type == "qnn_simple":
        # Simple classifier for quantum features
        input_size = kwargs.get("input_size")
        return SimpleClassifier(
            input_size=input_size,
            num_classes=config.get("num_classes", 10),
        )

    elif model_type == "random_nonlinear":
        return RandomNonlinearCNN(
            input_channels=kwargs.get("input_channels", 1),
            random_filters=config.get("quanv_filters", 25),
            conv1_filters=config.get("conv1_filters", 50),
            conv1_kernel=config.get("conv1_kernel", 5),
            conv2_filters=config.get("conv2_filters", 64),
            conv2_kernel=config.get("conv2_kernel", 5),
            fc1_units=config.get("fc1_units", 1024),
            dropout=config.get("dropout", 0.4),
            num_classes=config.get("num_classes", 10),
            seed=config.get("seed", 42),
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")