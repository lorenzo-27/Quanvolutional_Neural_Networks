"""
Dataset management for QNN experiments.
Supports MNIST, Fashion-MNIST, EMNIST, and CIFAR-10.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class DatasetManager:
    """Manages loading and preprocessing of datasets."""

    SUPPORTED_DATASETS = {
        "MNIST": torchvision.datasets.MNIST,
        "FashionMNIST": torchvision.datasets.FashionMNIST,
        "EMNIST": torchvision.datasets.EMNIST,
        "CIFAR10": torchvision.datasets.CIFAR10,
    }

    def __init__(
            self,
            dataset_name: str,
            data_path: str = "./data",
            normalize: bool = True,
    ):
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Dataset {dataset_name} not supported. "
                f"Choose from {list(self.SUPPORTED_DATASETS.keys())}"
            )

        self.dataset_name = dataset_name
        self.data_path = Path(data_path)
        self.normalize = normalize
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Set up transforms
        self.transform = self._get_transform()

    def _get_transform(self):
        """Get appropriate transform for the dataset."""
        transform_list = []

        # Convert to grayscale if needed (CIFAR-10)
        if self.dataset_name == "CIFAR10":
            transform_list.append(transforms.Grayscale())

        # Convert to tensor
        transform_list.append(transforms.ToTensor())

        # Normalize if requested
        if self.normalize:
            # Standard normalization: pixels in [0, 1]
            # Already done by ToTensor()
            pass

        return transforms.Compose(transform_list)

    def load_datasets(
            self,
            train_size: Optional[int] = None,
            test_size: Optional[int] = None,
    ) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """
        Load train and test datasets.

        Args:
            train_size: Limit training set size (None = full dataset)
            test_size: Limit test set size (None = full dataset)

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        dataset_class = self.SUPPORTED_DATASETS[self.dataset_name]

        # Special handling for EMNIST
        if self.dataset_name == "EMNIST":
            train_dataset = dataset_class(
                root=str(self.data_path),
                split="balanced",
                train=True,
                download=True,
                transform=self.transform
            )
            test_dataset = dataset_class(
                root=str(self.data_path),
                split="balanced",
                train=False,
                download=True,
                transform=self.transform
            )
        else:
            train_dataset = dataset_class(
                root=str(self.data_path),
                train=True,
                download=True,
                transform=self.transform
            )
            test_dataset = dataset_class(
                root=str(self.data_path),
                train=False,
                download=True,
                transform=self.transform
            )

        # Limit dataset sizes if requested
        if train_size is not None and train_size < len(train_dataset):
            indices = torch.randperm(len(train_dataset))[:train_size]
            train_dataset = Subset(train_dataset, indices)

        if test_size is not None and test_size < len(test_dataset):
            indices = torch.randperm(len(test_dataset))[:test_size]
            test_dataset = Subset(test_dataset, indices)

        return train_dataset, test_dataset

    def get_dataloaders(
            self,
            batch_size: int,
            train_size: Optional[int] = None,
            test_size: Optional[int] = None,
            num_workers: int = 4,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Get dataloaders for train and test sets.

        Args:
            batch_size: Batch size
            train_size: Limit training set size
            test_size: Limit test set size
            num_workers: Number of worker processes

        Returns:
            Tuple of (train_loader, test_loader)
        """
        train_dataset, test_dataset = self.load_datasets(train_size, test_size)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

        return train_loader, test_loader

    def get_numpy_data(
            self,
            train_size: Optional[int] = None,
            test_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get data as numpy arrays (useful for quantum preprocessing).

        Returns:
            Tuple of (train_images, train_labels, test_images, test_labels)
        """
        train_dataset, test_dataset = self.load_datasets(train_size, test_size)

        # Convert to numpy
        train_images = []
        train_labels = []
        for img, label in train_dataset:
            train_images.append(img.numpy())
            train_labels.append(label)

        test_images = []
        test_labels = []
        for img, label in test_dataset:
            test_images.append(img.numpy())
            test_labels.append(label)

        return (
            np.array(train_images),
            np.array(train_labels),
            np.array(test_images),
            np.array(test_labels)
        )

    @property
    def num_classes(self) -> int:
        """Get number of classes in the dataset."""
        if self.dataset_name in ["MNIST", "FashionMNIST", "CIFAR10"]:
            return 10
        elif self.dataset_name == "EMNIST":
            return 47  # balanced split
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    @property
    def image_shape(self) -> Tuple[int, int]:
        """Get image shape (H, W) for the dataset."""
        if self.dataset_name in ["MNIST", "FashionMNIST", "EMNIST"]:
            return (28, 28)
        elif self.dataset_name == "CIFAR10":
            return (32, 32)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")


def preprocess_for_quantum(
        images: np.ndarray,
        labels: np.ndarray,
        quanv_layer,
        cache_path: Optional[Path] = None,
        force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess images through quanvolutional layer.

    Args:
        images: Input images [N, C, H, W] or [N, H, W]
        labels: Labels [N]
        quanv_layer: QuanvolutionalLayer instance
        cache_path: Path to cache results
        force_recompute: Force recomputation even if cache exists

    Returns:
        Tuple of (preprocessed_images, labels)
    """
    cache_file = None
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_file = cache_path / "quantum_preprocessed.npz"

    # Check cache
    if cache_file and cache_file.exists() and not force_recompute:
        data = np.load(cache_file)
        return data["images"], data["labels"]

    # Preprocess images
    # Convert from [N, C, H, W] to [N, H, W]
    if images.ndim == 4:
        images = images[:, 0, :, :]

    preprocessed = quanv_layer.process_batch(images, show_progress=True)

    # Cache results
    if cache_file:
        np.savez(cache_file, images=preprocessed, labels=labels)

    return preprocessed, labels