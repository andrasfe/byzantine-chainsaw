"""Base configuration classes and enums."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

class DatasetType(Enum):
    MNIST = "mnist"
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"

class AttackType(Enum):
    LIE = "lie_attack"
    GAUSSIAN = "gaussian_attack"
    SIGN_FLIP = "sign_flip_attack"
    LABEL_FLIP = "label_flip_attack"

class DetectionMethod(Enum):
    STANDARD_MULTIKRUM = "standard_multikrum"
    CLASSICAL_RANDOM = "classical_random"
    CLASSICAL_IMPORTANCE = "classical_importance"
    QUANTUM_ZZ = "quantum_zz"
    QUANTUM_PAULI = "quantum_pauli"
    QUANTUM_HEISENBERG = "quantum_heisenberg"

class ProjectionMethod(Enum):
    RANDOM = "random"
    IMPORTANCE = "importance"
    PCA = "pca"

@dataclass
class FederatedConfig:
    """Configuration for federated learning"""
    num_honest_clients: int = 15
    num_byzantine_clients: int = 3
    num_rounds: int = 30
    local_epochs: int = 7
    batch_size: int = 32
    learning_rate: float = 0.07
    
    @property
    def total_clients(self) -> int:
        return self.num_honest_clients + self.num_byzantine_clients

@dataclass
class QuantumConfig:
    """Configuration for quantum computing"""
    num_qubits: int = 27
    num_shots: int = 1024
    backend_type: str = "aer_simulator"
    use_gpu: bool = True
    max_memory_mb: int = 20000
    batch_size: int = 3
    feature_map_reps: int = 2
    entanglement: str = "full"
    top_k_outcomes: int = 1000

@dataclass
class DetectionConfig:
    """Configuration for Byzantine detection"""
    detection_method: DetectionMethod = DetectionMethod.STANDARD_MULTIKRUM
    projection_method: ProjectionMethod = ProjectionMethod.IMPORTANCE
    projection_dim: int = 27
    detection_noise: float = 0.5
    
@dataclass
class AttackConfig:
    """Configuration for Byzantine attacks"""
    attack_type: AttackType = AttackType.LIE
    z_max_range: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    attack_strength: float = 1.0