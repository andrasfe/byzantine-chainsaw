# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a quantum-enhanced Byzantine detection system for federated learning that compares classical and quantum approaches for detecting malicious clients. The codebase contains both a modular implementation (`main.py` with subdirectories) and a standalone end-to-end script (`qe_multikrum.py`).

## Key Commands

### Running the main modular implementation
```bash
python main.py --config config.yaml               # Run full experiment
python main.py --config config.yaml --dry-run     # Validate configuration without execution
python main.py --config config.yaml --log-level DEBUG  # Debug mode
```

### Running the standalone script
```bash
python qe_multikrum.py  # Runs complete CIFAR-10 experiment with all detection methods
```

### Testing and validation
```bash
python -m pytest tests/ -v                        # Run all tests
python -m pytest tests/test_integration.py -v     # Run integration tests
python test_classical_embeddings.py               # Test classical embedding implementation
```

## Architecture

### Modular Implementation (`main.py` + subdirectories)
- **config/**: YAML configuration management
- **core/**: Abstract interfaces (`DetectionStrategy`, `AttackStrategy`, `FederatedDataManager`)
- **data/**: Dataset loading and federated data partitioning
- **models/**: CNN models with factory pattern
- **attacks/**: Byzantine attack implementations (lie attack, sign flipping, etc.)
- **detection/**: 
  - `classical/`: Multi-KRUM, random/importance projections
  - `quantum/`: ZZ, Pauli, Heisenberg feature maps with quantum embeddings
- **experiments/**: Orchestrates training rounds and detection
- **metrics/**: F1, Byzantine rejection, honest retention calculations
- **utils/**: Memory management for GPU/quantum computations

### Standalone Script (`qe_multikrum.py`)
Single-file implementation with all methods integrated:
- Includes classical embedding method (polynomial, trigonometric, exponential features)
- Non-linear feature maps (Polynomial, RBF, Fourier) with both random and importance projections
- Comprehensive visualization with multiple chart types
- Results saved as CSV tables and PNG plots

## Important Configuration

### Key parameters in `config.yaml`:
- `federated.num_byzantine_clients`: Number of malicious clients (default: 3)
- `quantum.num_qubits`: Dimension for quantum feature maps (default: 27)
- `quantum.batch_size`: Client batching for memory efficiency (default: 3)
- `detection.projection_dim`: Dimension for classical projections (default: 27)

### Hard-coded parameters in `qe_multikrum.py`:
- `PROJECTION_DIM = 27`: Matches quantum qubit count
- `LOCAL_EPOCHS = 7`: Training epochs per round
- `ROUNDS = 30`: Total federated learning rounds
- `TOP_K = 1000`: Sparse representation for quantum embeddings

## Detection Methods

All methods use Multi-KRUM for final client selection:

1. **Classical Standard**: Direct Multi-KRUM on raw gradients
2. **Classical + Projections**: Random or importance-weighted dimensionality reduction
3. **Classical + Non-linear Features**: Polynomial, RBF, or Fourier transformations
4. **Classical Embedding**: Custom embedding using polynomial + trigonometric + exponential features
5. **Quantum Methods**: ZZ, Pauli, or Heisenberg feature maps creating quantum state embeddings

## Memory Management

Critical for quantum simulations:
- Process clients in batches (default: 3 clients)
- Use sparse representation (top-1000 outcomes vs 2^27)
- Explicit garbage collection between rounds
- GPU memory limit: 20GB with automatic CPU fallback

## Visualization Output

The standalone script generates:
- `comprehensive_bar_chart_*.png`: 2x2 subplot comparing all methods
- `cifar10_tp_percentage_comparison_*.png`: TP rates for all methods
- `cifar10_grouped_tp_comparison_*.png`: Grouped comparison (best classical embedding only)
- `classical_methods_comparison_*.png`: Classical methods only
- `results_table_*.csv`: Complete results data
- `summary_table_*.csv`: Averaged performance metrics

## Common Tasks

### Adding a new detection method
1. Implement in `detection/classical/` or `detection/quantum/`
2. Register in appropriate factory
3. Add to `config.yaml` detection_methods list
4. For `qe_multikrum.py`: Add processing in main loop, update logging, add to visualization dicts

### Modifying attack parameters
- Edit `z_max_range` in `config.yaml` for modular version
- Change `z_max_options` in `generate_byzantine_update()` for standalone

### Adjusting for memory constraints
- Reduce `quantum.batch_size` (process fewer clients simultaneously)
- Lower `quantum.num_qubits` (smaller quantum circuits)
- Decrease `quantum.top_k_outcomes` (sparser representation)

## Important Notes

- The modular implementation uses dependency injection and factory patterns for extensibility
- The standalone script includes all visualizations and table generation
- Classical embedding in `qe_multikrum.py` mimics quantum behavior through non-linear transformations
- All non-linear feature maps now support both random and importance projections
- GPU acceleration available for both PyTorch (training) and Qiskit (quantum simulation)