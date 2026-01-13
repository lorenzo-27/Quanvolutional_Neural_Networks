"""
Quanvolutional layer implementation.
Main quantum feature extraction component.
"""

import pennylane as qml
import numpy as np
import torch
from typing import Tuple, List, Optional
from pathlib import Path
import pickle

from .encodings import BaseEncoding, get_encoding
from .circuits import BaseCircuit, get_circuit


class QuanvolutionalFilter:
    """
    Single quanvolutional filter that processes local image regions.

    Implements the quantum transformation: f_x = Q(u_x, e, q, d)
    where:
        - u_x: input data (2x2 region)
        - e: encoding function
        - q: quantum circuit
        - d: decoding (measurement)
    """

    def __init__(
            self,
            n_qubits: int,
            encoding: BaseEncoding,
            circuit: BaseCircuit,
            device: str = "default.qubit",
    ):
        self.n_qubits = n_qubits
        self.encoding = encoding
        self.circuit = circuit

        # Create quantum device
        self.dev = qml.device(device, wires=n_qubits)

        # Create quantum node
        self.qnode = qml.QNode(self._quantum_circuit, self.dev)

        # Cache for lookup table (optional optimization)
        self.lookup_table = {}

    def _quantum_circuit(self, data: np.ndarray) -> List[float]:
        """
        Define the quantum circuit.

        Args:
            data: Flattened input data

        Returns:
            List of expectation values (one per qubit)
        """
        wires = list(range(self.n_qubits))

        # Step 1: Encoding
        self.encoding(data, wires)

        # Step 2: Quantum computation
        self.circuit(wires)

        # Step 3: Measurement (decoding)
        # Return expectation value of Pauli-Z for each qubit
        return [qml.expval(qml.PauliZ(i)) for i in wires]

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Apply filter to input data.

        Args:
            data: Input data (flattened)

        Returns:
            Array of expectation values
        """
        # Convert to tuple for hashing (cache lookup)
        data_key = tuple(data.flatten())

        if data_key in self.lookup_table:
            return self.lookup_table[data_key]

        # Run quantum circuit
        result = self.qnode(data)
        result = np.array(result)

        # Cache result
        self.lookup_table[data_key] = result

        return result


class QuanvolutionalLayer:
    """
    Quanvolutional layer with multiple filters.
    Analogous to a classical convolutional layer.
    """

    def __init__(
            self,
            n_filters: int,
            kernel_size: int,
            stride: int,
            encoding_config: dict,
            circuit_config: dict,
            device: str = "default.qubit",
    ):
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_qubits = kernel_size ** 2

        # Create filters
        self.filters = []
        for i in range(n_filters):
            # Each filter has its own random circuit
            encoding = get_encoding(
                encoding_config["type"],
                self.n_qubits,
                **{k: v for k, v in encoding_config.items() if k != "type"}
            )

            # Create circuit with unique seed for each filter
            circuit_kwargs = {k: v for k, v in circuit_config.items() if k != "type"}
            if "random_seed" in circuit_kwargs:
                circuit_kwargs["seed"] = circuit_kwargs.pop("random_seed") + i

            circuit = get_circuit(
                circuit_config["type"],
                self.n_qubits,
                **circuit_kwargs
            )

            filter_obj = QuanvolutionalFilter(
                self.n_qubits,
                encoding,
                circuit,
                device
            )
            self.filters.append(filter_obj)

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process a single image through all filters.

        Args:
            image: Input image [H, W] or [H, W, C]

        Returns:
            Feature maps [H', W', n_filters * n_qubits]
        """
        if image.ndim == 3:
            image = image[:, :, 0]  # Use first channel only

        h, w = image.shape

        # Calculate output dimensions
        out_h = (h - self.kernel_size) // self.stride + 1
        out_w = (w - self.kernel_size) // self.stride + 1

        # Pre-extract all regions
        regions = []
        positions = []
        for i in range(0, h - self.kernel_size + 1, self.stride):
            for j in range(0, w - self.kernel_size + 1, self.stride):
                region = image[i:i + self.kernel_size, j:j + self.kernel_size]
                regions.append(region.flatten())
                positions.append((i // self.stride, j // self.stride))

        output = np.zeros((out_h, out_w, self.n_filters * self.n_qubits))

        # Process each filter
        for filter_idx, qfilter in enumerate(self.filters):
            for (out_i, out_j), region in zip(positions, regions):
                result = qfilter(region)
                start_ch = filter_idx * self.n_qubits
                end_ch = start_ch + self.n_qubits
                output[out_i, out_j, start_ch:end_ch] = result

        return output

    def process_batch(
            self,
            images: np.ndarray,
            show_progress: bool = True
    ) -> np.ndarray:
        """
        Process a batch of images.

        Args:
            images: Batch of images [N, H, W] or [N, H, W, C]
            show_progress: Whether to show progress bar

        Returns:
            Batch of feature maps [N, H', W', n_filters * n_qubits]
        """
        results = []

        iterator = enumerate(images)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=len(images), desc="Quanvolution")

        for idx, img in iterator:
            result = self.process_image(img)
            results.append(result)

        return np.array(results)

    def save_lookup_tables(self, path: Path):
        """Save all filter lookup tables for faster inference."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for i, qfilter in enumerate(self.filters):
            with open(path / f"filter_{i}_lookup.pkl", "wb") as f:
                pickle.dump(qfilter.lookup_table, f)

    def load_lookup_tables(self, path: Path):
        """Load pre-computed lookup tables."""
        path = Path(path)

        for i, qfilter in enumerate(self.filters):
            lookup_file = path / f"filter_{i}_lookup.pkl"
            if lookup_file.exists():
                with open(lookup_file, "rb") as f:
                    qfilter.lookup_table = pickle.load(f)