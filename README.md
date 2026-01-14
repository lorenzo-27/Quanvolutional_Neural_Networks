# Quanvolutional Neural Networks

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.9+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/PennyLane-0.44+-green.svg" alt="PennyLane">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

A research implementation of **Quanvolutional Neural Networks (QNNs)** for quantum machine learning, based on the foundational work by [Henderson et al. (2019)](https://arxiv.org/abs/1904.04767). This project explores hybrid quantum-classical neural networks that leverage quantum circuits as feature extractors for image classification tasks.

## 📖 Overview

Quanvolutional Neural Networks replace the first convolutional layer of a classical CNN with a **quantum convolutional layer** (quanvolution). The quantum layer applies parameterized quantum circuits to local regions of input images, producing quantum-enhanced feature maps that are then processed by classical neural network layers.

### Key Concepts

- **Quanvolution**: A quantum analog of convolution that applies quantum circuits to image patches
- **Quantum Feature Maps**: Transformations that encode classical data into quantum states
- **Hybrid Architecture**: Combines quantum preprocessing with classical neural networks

### Mathematical Framework

The quanvolutional transformation is defined as:

$$
f_x = Q(u_x, e, q, d)
$$

Where:
- $u_x$: Input data (e.g., 2×2 image patch)
- $e$: Encoding function (maps classical data to quantum states)
- $q$: Quantum circuit (parameterized unitary operations)
- $d$: Decoding (measurement in computational basis)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Image (28×28)                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               QUANTUM PREPROCESSING LAYER                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  For each 2×2 patch:                                │    │
│  │    1.  Encode pixels → Quantum states               │    │
│  │    2. Apply quantum circuit                         │    │
│  │    3. Measure expectation values                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                Output: 14×14×(n_filters × n_qubits)         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   CLASSICAL CNN LAYERS                      │
│  CONV1 (50 filters) → Pool → CONV2 (64 filters) → Pool      │
│                          │                                  │
│                          ▼                                  │
│              FC1 (1024) → Dropout → FC2 (10)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Classification Output
```

## ✨ Features

### Quantum Components

| Component | Options | Description |
|-----------|---------|-------------|
| **Encoding** | `threshold`, `angle`, `amplitude`, `dense` | Methods to encode classical data into quantum states |
| **Circuits** | `random`, `hardware_efficient`, `structured` | Quantum circuit architectures |
| **Entanglement** | `full`, `linear`, `circular` | Qubit connectivity patterns |

### Model Types

- **QNN (Quantum CNN)**: Full quanvolutional neural network
- **Classical CNN**: Standard CNN baseline for comparison
- **Random Nonlinear CNN**: Classical CNN with random fixed transformations

### Supported Datasets

- MNIST
- Fashion-MNIST
- EMNIST
- CIFAR-10

## 📁 Project Structure

```
Quanvolutional_Neural_Networks/
├── config/
│   └── config.yaml           # Main configuration file
├── experiments/
│   ├── run_experiments.py    # Main experiment runner
│   ├── compare_models.py     # Model comparison utilities
│   └── sweep_experiment.py   # Hyperparameter sweeps
├── src/
│   ├── data/
│   │   └── dataset. py       # Dataset loading and preprocessing
│   ├── models/
│   │   └── qnn.py            # Neural network architectures
│   ├── quantum/
│   │   ├── circuits.py       # Quantum circuit implementations
│   │   ├── encodings.py      # Data encoding methods
│   │   └── quanvolution.py   # Quanvolutional layer
│   ├── training/
│   │   └── trainer.py        # Training loop and utilities
│   ├── utils/
│   │   └── helpers.py        # Utility functions
│   └── visualization/
│       └── plots.py          # Visualization tools
├── pyproject.toml            # Project dependencies
└── uv.lock                   # Lock file for reproducibility
```

## 🚀 Installation

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/lorenzo-27/Quanvolutional_Neural_Networks.git
cd Quanvolutional_Neural_Networks

# Install dependencies
uv sync
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/lorenzo-27/Quanvolutional_Neural_Networks. git
cd Quanvolutional_Neural_Networks

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e . 
```

## ⚙️ Configuration

All experiment parameters are controlled via `config/config.yaml`:

```yaml
# Experiment settings
experiment: 
  name: "qnn_baseline"
  seed: 42
  device: "auto"  # auto, cuda, cpu

# Dataset configuration
dataset:
  name: "MNIST"  # Options: MNIST, FashionMNIST, EMNIST, CIFAR10
  train_size: 120
  test_size: 20

# Quantum circuit configuration
quantum:
  n_qubits:  4         # For 2x2 filters
  kernel_size: 2       # 2x2 convolution window
  stride: 2
  
  encoding:
    type: "threshold"   # Options: threshold, angle, amplitude, dense
    
  circuit:
    type:  "random"     # Options: random, structured, hardware_efficient
    n_layers: 1
    entanglement: "full"

# Model configuration
model: 
  type: "qnn"          # Options: qnn, classical, random_nonlinear
  quanv_filters: 25    # Number of quantum filters
  conv1_filters: 50
  conv2_filters: 64
  fc1_units: 1024
  dropout:  0.4

# Training configuration
training:
  epochs: 30
  batch_size: 128
  learning_rate: 0.001
  optimizer: "adam"
```

## 🎮 Usage

### Running a Single Experiment

```bash
# Run with default configuration
uv run python experiments/run_experiments.py

# Run with custom configuration
uv run python experiments/run_experiments.py --config path/to/config. yaml
```

### Comparing Models

```bash
# Compare QNN vs Classical CNN vs Random Nonlinear
uv run python experiments/compare_models.py
```

### Hyperparameter Sweeps

```bash
# Run hyperparameter sweep experiments
uv run python experiments/sweep_experiment.py
```

### Example:  Training a QNN

```python
from src.data.dataset import DatasetManager
from src.quantum.quanvolution import QuanvolutionalLayer
from src. models.qnn import get_model
from src.training.trainer import Trainer

# Load dataset
dataset_manager = DatasetManager(dataset_name="MNIST")
train_loader, test_loader = dataset_manager. get_dataloaders(batch_size=128)

# Create quanvolutional layer
quanv_layer = QuanvolutionalLayer(
    n_filters=25,
    kernel_size=2,
    stride=2,
    encoding_config={"type": "threshold"},
    circuit_config={"type": "random", "n_layers": 1}
)

# Create and train model
model = get_model("qnn", config, quantum_channels=100)
trainer = Trainer(model=model, config=config, device="cuda")
trainer.train(train_loader, test_loader, num_epochs=30)
```

## 📊 Experiment Tracking

This project supports [Weights & Biases](https://wandb.ai/) for experiment tracking: 

```yaml
# Enable in config.yaml
wandb:
  enabled: true
  project: "quanvolutional-nn"
  entity: "your-username"
  tags: ["baseline", "mnist"]
```

## 🔬 Quantum Circuit Types

### Random Circuit
Based on Henderson et al. (2019), generates random gates from a specified gate set with random parameters: 

```python
RandomCircuit(
    n_qubits=4,
    n_layers=1,
    gate_set=["RX", "RY", "RZ", "CNOT"],
    connection_prob=0.5
)
```

### Hardware-Efficient Circuit
Optimized for near-term quantum devices with alternating rotation and entanglement layers:

```python
HardwareEfficientCircuit(
    n_qubits=4,
    n_layers=2,
    entanglement="linear"  # or "circular", "full"
)
```

### Structured Circuit
Implements specific quantum feature maps like ZZ or Pauli: 

```python
StructuredCircuit(
    n_qubits=4,
    n_layers=1,
    feature_map="zz"  # or "pauli"
)
```

## 📈 Encoding Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| **Threshold** | Binary encoding based on pixel threshold | Simple, fast preprocessing |
| **Angle** | Encodes values as rotation angles | Continuous data encoding |
| **Amplitude** | Amplitude encoding of normalized data | Dense quantum states |
| **Dense** | Multiple rotation gates per qubit | Maximum expressibility |

## 📚 References

1. Henderson, M., Shakya, S., Pradhan, S., & Cook, T. (2019). **Quanvolutional Neural Networks:  Powering Image Recognition with Quantum Circuits**. arXiv: 1904.04767

2. Farhi, E., & Neven, H. (2018). **Classification with Quantum Neural Networks on Near Term Processors**. arXiv:1802.06002

3. PennyLane Documentation:  [Quanvolutional Neural Networks Tutorial](https://pennylane.ai/qml/demos/tutorial_quanvolution.html)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.  For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the <a href="https://github.com/lorenzo-27/Quanvolutional_Neural_Networks/blob/master/LICENSE" target="_blank">MIT</a> License.

---

<p align="center">
  <i>Exploring the intersection of quantum computing and machine learning</i>
</p>
