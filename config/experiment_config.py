"""Experiment configuration management."""

from dataclasses import dataclass
from typing import Dict, Any, List
import yaml
from pathlib import Path

from .base_config import (
    FederatedConfig, QuantumConfig, DetectionConfig, AttackConfig,
    DatasetType, DetectionMethod, AttackType
)

@dataclass
class ExperimentConfig:
    """Main experiment configuration"""
    experiment_name: str
    dataset: DatasetType
    federated: FederatedConfig
    quantum: QuantumConfig
    detection: DetectionConfig
    attack: AttackConfig
    detection_methods: List[DetectionMethod] = None
    seed: int = 42
    device: str = "auto"  # auto, cuda, cpu
    log_level: str = "INFO"
    save_results: bool = True
    output_dir: str = "./results"
    
    def __post_init__(self):
        """Set default detection methods if not specified"""
        if self.detection_methods is None:
            self.detection_methods = [
                DetectionMethod.STANDARD_MULTIKRUM,
                DetectionMethod.CLASSICAL_RANDOM,
                DetectionMethod.CLASSICAL_IMPORTANCE,
                DetectionMethod.QUANTUM_ZZ,
                DetectionMethod.QUANTUM_PAULI,
                DetectionMethod.QUANTUM_HEISENBERG
            ]
    
    @classmethod
    def from_yaml(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file"""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
            
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create configuration from dictionary"""
        # Parse nested configurations
        federated_dict = config_dict.get('federated', {})
        quantum_dict = config_dict.get('quantum', {})
        detection_dict = config_dict.get('detection', {})
        attack_dict = config_dict.get('attack', {})
        
        # Parse detection methods
        detection_methods = None
        if 'detection_methods' in detection_dict:
            method_names = detection_dict.pop('detection_methods')
            detection_methods = [DetectionMethod(method) for method in method_names]
        
        # Parse attack type in attack_dict
        if 'attack_type' in attack_dict:
            attack_dict['attack_type'] = AttackType(attack_dict['attack_type'])
        
        # Parse detection method in detection_dict
        if 'detection_method' in detection_dict:
            detection_dict['detection_method'] = DetectionMethod(detection_dict['detection_method'])
        if 'projection_method' in detection_dict:
            from .base_config import ProjectionMethod
            detection_dict['projection_method'] = ProjectionMethod(detection_dict['projection_method'])
        
        return cls(
            experiment_name=config_dict.get('experiment_name', 'quantum_byzantine_detection'),
            dataset=DatasetType(config_dict.get('dataset', 'cifar10')),
            federated=FederatedConfig(**federated_dict),
            quantum=QuantumConfig(**quantum_dict),
            detection=DetectionConfig(**detection_dict),
            attack=AttackConfig(**attack_dict),
            detection_methods=detection_methods,
            seed=config_dict.get('seed', 42),
            device=config_dict.get('device', 'auto'),
            log_level=config_dict.get('log_level', 'INFO'),
            save_results=config_dict.get('save_results', True),
            output_dir=config_dict.get('output_dir', './results')
        )
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file"""
        config_dict = {
            'experiment_name': self.experiment_name,
            'dataset': self.dataset.value,
            'federated': {
                'num_honest_clients': self.federated.num_honest_clients,
                'num_byzantine_clients': self.federated.num_byzantine_clients,
                'num_rounds': self.federated.num_rounds,
                'local_epochs': self.federated.local_epochs,
                'batch_size': self.federated.batch_size,
                'learning_rate': self.federated.learning_rate
            },
            'quantum': {
                'num_qubits': self.quantum.num_qubits,
                'num_shots': self.quantum.num_shots,
                'backend_type': self.quantum.backend_type,
                'use_gpu': self.quantum.use_gpu,
                'max_memory_mb': self.quantum.max_memory_mb,
                'batch_size': self.quantum.batch_size,
                'feature_map_reps': self.quantum.feature_map_reps,
                'entanglement': self.quantum.entanglement,
                'top_k_outcomes': self.quantum.top_k_outcomes
            },
            'detection': {
                'detection_method': self.detection.detection_method.value,
                'projection_method': self.detection.projection_method.value,
                'projection_dim': self.detection.projection_dim,
                'detection_noise': self.detection.detection_noise,
                'detection_methods': [method.value for method in self.detection_methods]
            },
            'attack': {
                'attack_type': self.attack.attack_type.value,
                'z_max_range': self.attack.z_max_range,
                'attack_strength': self.attack.attack_strength
            },
            'seed': self.seed,
            'device': self.device,
            'log_level': self.log_level,
            'save_results': self.save_results,
            'output_dir': self.output_dir
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)