# Quantum-Enhanced Byzantine Detection in Federated Learning

A comprehensive implementation comparing classical and quantum approaches for detecting Byzantine (malicious) clients in federated learning systems using CIFAR-10 dataset.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.1.1-green.svg)](https://qiskit.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🔬 Overview

This project implements a modular, extensible system for comparing classical and quantum approaches to Byzantine fault detection in federated learning. The system supports multiple detection methods including:

- **Classical Methods**: Standard Multi-KRUM, Random Projection, Importance-Weighted Projection
- **Quantum Methods**: ZZ Feature Maps, Pauli Feature Maps, Heisenberg Feature Maps with quantum embeddings

### Key Features

- 🧮 **Modular Architecture**: Clean separation of concerns with dependency injection
- ⚡ **GPU Acceleration**: CUDA support for both PyTorch training and Qiskit quantum simulation
- 🔧 **Configurable**: YAML-based configuration for easy experimentation  
- 📊 **Comprehensive Metrics**: F1-score, Byzantine rejection rate, honest retention
- 🧪 **Memory Optimized**: Efficient quantum embedding computation with batching
- 🔄 **Reproducible**: Seed-based reproducibility for consistent results

## 🏗️ Architecture

```
qe_byzantine_detection/
├── config/                    # Configuration management
├── core/                      # Abstract interfaces and exceptions  
├── data/                      # Dataset loading and federated data management
├── models/                    # Neural network models with factory pattern
├── attacks/                   # Byzantine attack implementations
├── detection/                 # Detection algorithms
│   ├── classical/             # Classical Multi-KRUM and projections
│   └── quantum/               # Quantum feature maps and embeddings
├── experiments/               # Experiment orchestration
├── metrics/                   # Performance evaluation
├── utils/                     # Memory management and utilities
└── main.py                    # Entry point
```

## 📊 Detection Methods

### Classical Approaches

1. **Standard Multi-KRUM**: Direct application on raw gradient updates
2. **Multi-KRUM + Random Projection**: Gaussian random projection to reduce dimensionality
3. **Multi-KRUM + Importance-Weighted Projection**: Variance-based projection weighting

### Quantum Approaches

1. **ZZ Feature Maps**: Entangling gates with Z rotations
2. **Pauli Feature Maps**: X, Y, Z, XY, YZ Pauli terms with full entanglement  
3. **Heisenberg Feature Maps**: XX, YY, ZZ interaction terms

Each quantum method uses:
- Memory-efficient sparse representation (top-1000 measurement outcomes)
- Parameterized quantum circuits with 2 repetitions
- GPU-accelerated quantum simulation when available

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional but recommended)
- 16GB+ RAM recommended for quantum simulations

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd qe_byzantine_detection
```

2. **Create and activate virtual environment**:
```bash
python -m venv qe_byzantine_venv
source qe_byzantine_venv/bin/activate  # Linux/Mac
# qe_byzantine_venv\\Scripts\\activate  # Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Verify installation**:
```bash
make validate-all  # Run comprehensive validation
```

### Running Experiments

1. **Basic experiment with default configuration**:
```bash
python main.py --config config.yaml
```

2. **Dry run to validate configuration**:
```bash
python main.py --config config.yaml --dry-run
```

3. **Debug mode with verbose logging**:
```bash
python main.py --config config.yaml --log-level DEBUG
```

## ⚙️ Configuration

The system uses YAML configuration files. Key sections:

### Federated Learning Setup
```yaml
federated:
  num_honest_clients: 15      # Number of honest participants
  num_byzantine_clients: 3    # Number of Byzantine participants  
  num_rounds: 30              # Total training rounds
  local_epochs: 7             # Local training epochs per round
  batch_size: 32              # Training batch size
  learning_rate: 0.07         # SGD learning rate
```

### Quantum Configuration
```yaml
quantum:
  num_qubits: 27              # Quantum feature map dimension
  num_shots: 1024             # Measurement shots per circuit
  use_gpu: true               # Enable GPU acceleration
  max_memory_mb: 20000        # GPU memory limit (MB)
  batch_size: 3               # Client batch size for quantum computation
  top_k_outcomes: 1000        # Top-K measurement outcomes to store
```

### Detection Methods
```yaml
detection:
  detection_methods:
    - "standard_multikrum"     # Classical baseline
    - "classical_random"       # Classical + random projection
    - "classical_importance"   # Classical + importance weighting
    - "quantum_zz"            # Quantum ZZ feature maps
    - "quantum_pauli"         # Quantum Pauli feature maps
    - "quantum_heisenberg"    # Quantum Heisenberg feature maps
```

### Byzantine Attacks
```yaml
attack:
  attack_type: "lie_attack"              # Attack strategy
  z_max_range: [0.0, 0.1, ..., 1.0]     # Attack strength parameters
  attack_strength: 1.0                   # Global attack multiplier
```

## 🧪 Experimental Results

The system evaluates performance using several metrics:

- **True Positives (TP)**: Byzantine clients correctly rejected
- **False Positives (FP)**: Honest clients incorrectly rejected  
- **Byzantine Rejection Rate**: TP/(TP+FN) - recall
- **Honest Retention**: TN/(TN+FP) - specificity
- **F1 Score**: Harmonic mean of precision and recall
- **Test Accuracy**: Final model performance

### Expected Performance

Based on the research, quantum methods may demonstrate:
- Enhanced detection of sophisticated Byzantine attacks
- Better performance in high-dimensional gradient spaces
- Improved robustness to noise and parameter variations

## 🔬 Algorithm Details

### Multi-KRUM Algorithm

Multi-KRUM selects clients based on distance to nearest neighbors:

1. Compute pairwise distances between all client updates
2. For each client, calculate score as sum of distances to k nearest neighbors
3. Select n-f clients with lowest scores (where f = number of Byzantine clients)

**Parameters**: k = n - f - 2 (where n = total clients, f = Byzantine clients)

### Quantum Feature Encoding

Quantum methods encode classical gradient vectors into quantum states:

1. **Data Preprocessing**: Normalize gradient vectors for quantum encoding
2. **Circuit Construction**: Build parameterized quantum circuits with chosen feature map
3. **Measurement**: Execute circuits and collect measurement statistics
4. **Embedding**: Create sparse representation using top-K measurement outcomes
5. **Distance Computation**: Apply Multi-KRUM on quantum embedding space

### Memory Optimization

For efficient quantum computation:
- **Batch Processing**: Process clients in small batches (default: 3)
- **Sparse Representation**: Store only top-1000 measurement outcomes vs 2^n
- **Memory Cleanup**: Explicit garbage collection and CUDA cache clearing
- **Progressive Processing**: Sequential client processing with intermediate cleanup

## 📈 Performance Optimization

### GPU Acceleration
- **PyTorch**: Automatic CUDA utilization for model training
- **Qiskit**: GPU-based quantum simulation with cuQuantum support
- **Memory Management**: 20GB GPU memory limit with automatic fallback

### Scalability Features
- **Configurable Batch Sizes**: Adjust based on available memory
- **Projection Dimensionality**: Reduce quantum circuit complexity  
- **Method Selection**: Enable/disable specific detection methods
- **Parallel Processing**: Batch operations where possible

## 🛠️ Development

### Build System

The project includes a comprehensive Makefile:

```bash
make validate-all    # Complete validation pipeline
make check-syntax    # Python syntax validation
make test-imports    # Import structure validation  
make lint           # Code quality checks
make clean          # Cleanup build artifacts
```

### Code Quality

- **Type Hints**: Comprehensive type annotations
- **Abstract Interfaces**: Clean separation of concerns
- **Factory Patterns**: Extensible component creation
- **Error Handling**: Robust exception handling with fallbacks
- **Documentation**: Comprehensive docstrings and comments

### Testing

The modular architecture supports comprehensive testing:
- **Unit Tests**: Individual component validation
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Memory and runtime validation

## 🔧 Extending the System

### Adding New Detection Methods

1. **Implement Interface**: Create class inheriting from `DetectionStrategy`
2. **Register Factory**: Add to appropriate factory class
3. **Update Config**: Add method to configuration enum
4. **Test Integration**: Ensure proper integration in experiment runner

### Adding New Attack Strategies

1. **Implement Interface**: Create class inheriting from `AttackStrategy`  
2. **Register Factory**: Add to `AttackFactory`
3. **Configuration**: Add to `AttackType` enum
4. **Validation**: Test against existing detection methods

### Adding New Quantum Feature Maps

1. **Extend Builder**: Add new feature map type to `QuantumFeatureMapBuilder`
2. **Update Detection**: Ensure compatibility with quantum detection pipeline
3. **Memory Testing**: Validate memory efficiency for new circuit designs

## 🐛 Troubleshooting

### Common Issues

**Memory Errors**:
- Reduce `quantum.batch_size` in configuration
- Lower `quantum.num_qubits` for smaller circuits
- Decrease `quantum.top_k_outcomes` for reduced storage

**CUDA Issues**:
- Set `quantum.use_gpu: false` for CPU-only mode
- Check CUDA driver compatibility
- Verify cuQuantum installation for quantum GPU acceleration

**Import Errors**:
- Run `make validate-all` to check dependencies
- Ensure virtual environment activation
- Verify Qiskit version compatibility

**Performance Issues**:
- Enable GPU acceleration where possible
- Reduce number of detection methods for faster execution
- Lower `quantum.num_shots` for faster quantum computation

### Logging and Debugging

- Set `log_level: DEBUG` for detailed execution logs
- Check `results/experiment.log` for complete execution history
- Use `--dry-run` flag to validate configuration without execution

## 📚 References

This implementation is based on research in quantum-enhanced Byzantine detection for federated learning, combining classical Multi-KRUM algorithms with quantum feature mapping techniques for improved malicious client identification.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📞 Support

For questions and support:
- 📧 Email: [your-email@domain.com]
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/repo/issues)
- 📖 Documentation: [Project Wiki](https://github.com/your-username/repo/wiki)

---

**Built with ❤️ for advancing quantum-classical hybrid approaches to distributed system security.**