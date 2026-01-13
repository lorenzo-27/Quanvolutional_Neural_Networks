"""
Quantum encoding schemes for classical data.
Implements various methods to encode classical pixel values into quantum states.
"""

import pennylane as qml
import numpy as np
from typing import List, Callable
from abc import ABC, abstractmethod


class BaseEncoding(ABC):
    """Abstract base class for quantum encodings."""

    def __init__(self, n_qubits: int, **kwargs):
        self.n_qubits = n_qubits

    @abstractmethod
    def encode(self, data: np.ndarray, wires: List[int]) -> None:
        """
        Encode classical data into quantum state.

        Args:
            data: Classical data to encode (flattened array)
            wires: Qubit indices to use
        """
        pass

    def __call__(self, data: np.ndarray, wires: List[int]) -> None:
        return self.encode(data, wires)


class ThresholdEncoding(BaseEncoding):
    """
    Threshold encoding: binary encoding based on threshold value.
    Pixels > threshold → |1⟩, pixels ≤ threshold → |0⟩

    This is the encoding used in Henderson et al. 2019.
    """

    def __init__(self, n_qubits: int, threshold: float = 0.0, **kwargs):
        super().__init__(n_qubits)
        self.threshold = threshold

    def encode(self, data: np.ndarray, wires: List[int]) -> None:
        """Encode data using threshold."""
        assert len(data) == len(wires), "Data size must match number of wires"

        for i, wire in enumerate(wires):
            if data[i] > self.threshold:
                qml.PauliX(wires=wire)


class AngleEncoding(BaseEncoding):
    """
    Angle encoding: encode pixel intensity as rotation angles.
    More expressive than threshold encoding as it preserves grayscale information.

    Uses RY rotations: RY(π * pixel_value)
    """

    def __init__(self, n_qubits: int, scale_factor: float = np.pi, **kwargs):
        super().__init__(n_qubits)
        self.scale_factor = scale_factor

    def encode(self, data: np.ndarray, wires: List[int]) -> None:
        """Encode data using angle encoding."""
        assert len(data) == len(wires), "Data size must match number of wires"

        for i, wire in enumerate(wires):
            qml.RY(self.scale_factor * data[i], wires=wire)


class AmplitudeEncoding(BaseEncoding):
    """
    Amplitude encoding: encode data into amplitudes of quantum state.
    Most efficient but requires normalization and padding.

    For n qubits, can encode 2^n values.
    """

    def __init__(self, n_qubits: int, normalize: bool = True, **kwargs):
        super().__init__(n_qubits)
        self.normalize = normalize

    def encode(self, data: np.ndarray, wires: List[int]) -> None:
        """Encode data using amplitude encoding."""
        # Pad data to 2^n if necessary
        target_size = 2 ** len(wires)
        if len(data) < target_size:
            data = np.pad(data, (0, target_size - len(data)))
        elif len(data) > target_size:
            data = data[:target_size]

        # Normalize if required
        if self.normalize:
            norm = np.linalg.norm(data)
            if norm > 0:
                data = data / norm

        qml.AmplitudeEmbedding(features=data, wires=wires, normalize=False)


class DenseAngleEncoding(BaseEncoding):
    """
    Dense angle encoding: uses multiple rotation gates per qubit.
    Applies RX, RY, RZ rotations to create richer feature space.
    """

    def __init__(self, n_qubits: int, scale_factor: float = np.pi, **kwargs):
        super().__init__(n_qubits)
        self.scale_factor = scale_factor

    def encode(self, data: np.ndarray, wires: List[int]) -> None:
        """Encode data using dense angle encoding."""
        assert len(data) == len(wires), "Data size must match number of wires"

        for i, wire in enumerate(wires):
            angle = self.scale_factor * data[i]
            qml.RY(angle, wires=wire)
            qml.RZ(angle / 2, wires=wire)  # Additional rotation


def get_encoding(encoding_type: str, n_qubits: int, **kwargs) -> BaseEncoding:
    """
    Factory function to get encoding by name.

    Args:
        encoding_type: Type of encoding (threshold, angle, amplitude, dense)
        n_qubits: Number of qubits
        **kwargs: Additional parameters for specific encodings

    Returns:
        Encoding instance
    """
    encodings = {
        "threshold": ThresholdEncoding,
        "angle": AngleEncoding,
        "amplitude": AmplitudeEncoding,
        "dense": DenseAngleEncoding,
    }

    if encoding_type not in encodings:
        raise ValueError(
            f"Unknown encoding type: {encoding_type}. "
            f"Available: {list(encodings.keys())}"
        )

    return encodings[encoding_type](n_qubits, **kwargs)