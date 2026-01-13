"""
Quantum circuit implementations for quanvolutional filters.
Supports random circuits (paper baseline) and structured ansatzes.
"""

import pennylane as qml
import numpy as np
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod


class BaseCircuit(ABC):
    """Abstract base class for quantum circuits."""

    def __init__(self, n_qubits: int, **kwargs):
        self.n_qubits = n_qubits
        self.params = None

    @abstractmethod
    def build_circuit(self, wires: List[int]) -> None:
        """Build the quantum circuit."""
        pass

    def __call__(self, wires: List[int]) -> None:
        return self.build_circuit(wires)


class RandomCircuit(BaseCircuit):
    """
    Random quantum circuit as described in Henderson et al. 2019.

    Generates random gates from a specified gate set with random parameters.
    Uses connection probability for 2-qubit gates.
    """

    def __init__(
            self,
            n_qubits: int,
            n_layers: int = 1,
            gate_set: Optional[List[str]] = None,
            seed: Optional[int] = None,
            connection_prob: float = 0.5,
            **kwargs,
    ):
        super().__init__(n_qubits)
        self.n_layers = n_layers
        self.gate_set = gate_set or ["RX", "RY", "RZ", "CNOT"]
        self.connection_prob = connection_prob

        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Generate random parameters
        self._generate_params()

    def _generate_params(self):
        """Generate random circuit parameters."""
        # Number of 1-qubit gates per layer
        n_single_gates = np.random.randint(0, 2 * self.n_qubits ** 2 + 1)

        # Random parameters for rotation gates
        self.rotation_params = np.random.uniform(
            0, 2 * np.pi, size=(self.n_layers, n_single_gates)
        )

        # Random qubit targets for single-qubit gates
        self.single_qubit_targets = np.random.randint(
            0, self.n_qubits, size=(self.n_layers, n_single_gates)
        )

        # Random gate types
        rotation_gates = [g for g in self.gate_set if g in ["RX", "RY", "RZ"]]
        self.gate_types = np.random.choice(
            rotation_gates, size=(self.n_layers, n_single_gates)
        )

        # 2-qubit gate connections
        self.connections = []
        for _ in range(self.n_layers):
            layer_connections = []
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    if np.random.random() < self.connection_prob:
                        layer_connections.append((i, j))
            self.connections.append(layer_connections)

    def build_circuit(self, wires: List[int]) -> None:
        """Build the random circuit."""
        for layer in range(self.n_layers):
            # Apply single-qubit gates
            for i in range(len(self.rotation_params[layer])):
                gate_type = self.gate_types[layer, i]
                target = self.single_qubit_targets[layer, i]
                param = self.rotation_params[layer, i]

                if gate_type == "RX":
                    qml.RX(param, wires=wires[target])
                elif gate_type == "RY":
                    qml.RY(param, wires=wires[target])
                elif gate_type == "RZ":
                    qml.RZ(param, wires=wires[target])

            # Apply 2-qubit gates
            for i, j in self.connections[layer]:
                if "CNOT" in self.gate_set:
                    qml.CNOT(wires=[wires[i], wires[j]])


class HardwareEfficientCircuit(BaseCircuit):
    """
    Hardware-efficient ansatz circuit.
    Alternates single-qubit rotations with entangling gates.
    """

    def __init__(
            self,
            n_qubits: int,
            n_layers: int = 1,
            entanglement: str = "linear",
            seed: Optional[int] = None,
            **kwargs,
    ):
        super().__init__(n_qubits)
        self.n_layers = n_layers
        self.entanglement = entanglement

        if seed is not None:
            np.random.seed(seed)

        # Generate random parameters
        self.params = np.random.uniform(
            0, 2 * np.pi, size=(n_layers, n_qubits, 3)
        )

    def build_circuit(self, wires: List[int]) -> None:
        """Build hardware-efficient circuit."""
        for layer in range(self.n_layers):
            # Rotation layer
            for i, wire in enumerate(wires):
                qml.RY(self.params[layer, i, 0], wires=wire)
                qml.RZ(self.params[layer, i, 1], wires=wire)

            # Entangling layer
            if self.entanglement == "linear":
                for i in range(len(wires) - 1):
                    qml.CNOT(wires=[wires[i], wires[i + 1]])
            elif self.entanglement == "circular":
                for i in range(len(wires)):
                    qml.CNOT(wires=[wires[i], wires[(i + 1) % len(wires)]])
            elif self.entanglement == "full":
                for i in range(len(wires)):
                    for j in range(i + 1, len(wires)):
                        qml.CNOT(wires=[wires[i], wires[j]])


class StructuredCircuit(BaseCircuit):
    """
    Structured circuit with specific entanglement patterns.
    Useful for investigating specific quantum feature maps.
    """

    def __init__(
            self,
            n_qubits: int,
            n_layers: int = 1,
            feature_map: str = "zz",  # Options: zz, pauli, iqp
            seed: Optional[int] = None,
            **kwargs,
    ):
        super().__init__(n_qubits)
        self.n_layers = n_layers
        self.feature_map = feature_map

        if seed is not None:
            np.random.seed(seed)

        self.params = np.random.uniform(
            0, 2 * np.pi, size=(n_layers, n_qubits)
        )

    def build_circuit(self, wires: List[int]) -> None:
        """Build structured circuit."""
        if self.feature_map == "zz":
            self._build_zz_feature_map(wires)
        elif self.feature_map == "pauli":
            self._build_pauli_feature_map(wires)
        else:
            raise ValueError(f"Unknown feature map: {self.feature_map}")

    def _build_zz_feature_map(self, wires: List[int]) -> None:
        """ZZ feature map for edge detection."""
        for layer in range(self.n_layers):
            # Hadamard layer
            for wire in wires:
                qml.Hadamard(wires=wire)

            # ZZ interactions
            for i in range(len(wires) - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
                qml.RZ(self.params[layer, i], wires=wires[i + 1])
                qml.CNOT(wires=[wires[i], wires[i + 1]])

    def _build_pauli_feature_map(self, wires: List[int]) -> None:
        """Pauli feature map."""
        for layer in range(self.n_layers):
            for i, wire in enumerate(wires):
                qml.Hadamard(wires=wire)
                qml.RZ(self.params[layer, i], wires=wire)


def get_circuit(circuit_type: str, n_qubits: int, **kwargs) -> BaseCircuit:
    """
    Factory function to get circuit by type.

    Args:
        circuit_type: Type of circuit (random, hardware_efficient, structured)
        n_qubits: Number of qubits
        **kwargs: Additional parameters

    Returns:
        Circuit instance
    """
    circuits = {
        "random": RandomCircuit,
        "hardware_efficient": HardwareEfficientCircuit,
        "structured": StructuredCircuit,
    }

    if circuit_type not in circuits:
        raise ValueError(
            f"Unknown circuit type: {circuit_type}. "
            f"Available: {list(circuits.keys())}"
        )

    return circuits[circuit_type](n_qubits, **kwargs)