"""Tests for configuration management."""

import pytest
import tempfile
import yaml
from pathlib import Path

from config.base_config import (
    FederatedConfig, QuantumConfig, DetectionConfig, AttackConfig,
    DatasetType, AttackType, DetectionMethod, ProjectionMethod
)
from config.experiment_config import ExperimentConfig

class TestBaseConfigs:
    """Test base configuration classes"""
    
    def test_federated_config_defaults(self):
        """Test FederatedConfig with default values"""
        config = FederatedConfig()
        assert config.num_honest_clients == 15
        assert config.num_byzantine_clients == 3
        assert config.total_clients == 18
        
    def test_federated_config_custom(self):
        """Test FederatedConfig with custom values"""
        config = FederatedConfig(num_honest_clients=10, num_byzantine_clients=2)
        assert config.total_clients == 12
        
    def test_quantum_config_defaults(self):
        """Test QuantumConfig with default values"""
        config = QuantumConfig()
        assert config.num_qubits == 27
        assert config.num_shots == 1024
        assert config.use_gpu == True
        
    def test_detection_config_defaults(self):
        """Test DetectionConfig with default values"""
        config = DetectionConfig()
        assert config.detection_method == DetectionMethod.STANDARD_MULTIKRUM
        assert config.projection_method == ProjectionMethod.IMPORTANCE
        
    def test_attack_config_defaults(self):
        """Test AttackConfig with default values"""
        config = AttackConfig()
        assert config.attack_type == AttackType.LIE
        assert len(config.z_max_range) == 11

class TestExperimentConfig:
    """Test experiment configuration management"""
    
    def test_experiment_config_creation(self):
        """Test creating ExperimentConfig"""
        config = ExperimentConfig(
            experiment_name="test_experiment",
            dataset=DatasetType.CIFAR10,
            federated=FederatedConfig(),
            quantum=QuantumConfig(),
            detection=DetectionConfig(),
            attack=AttackConfig()
        )
        assert config.experiment_name == "test_experiment"
        assert config.dataset == DatasetType.CIFAR10
        assert len(config.detection_methods) == 6  # Default methods
        
    def test_yaml_roundtrip(self):
        """Test saving and loading YAML configuration"""
        config = ExperimentConfig(
            experiment_name="test_experiment",
            dataset=DatasetType.CIFAR10,
            federated=FederatedConfig(num_honest_clients=10),
            quantum=QuantumConfig(num_qubits=20),
            detection=DetectionConfig(),
            attack=AttackConfig()
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.to_yaml(f.name)
            
            # Load and verify
            loaded_config = ExperimentConfig.from_yaml(f.name)
            assert loaded_config.experiment_name == "test_experiment"
            assert loaded_config.federated.num_honest_clients == 10
            assert loaded_config.quantum.num_qubits == 20
            
        Path(f.name).unlink()  # Clean up
        
    def test_from_dict(self):
        """Test creating config from dictionary"""
        config_dict = {
            'experiment_name': 'test_from_dict',
            'dataset': 'mnist',
            'federated': {'num_honest_clients': 8},
            'quantum': {'num_qubits': 15},
            'detection': {'projection_dim': 20},
            'attack': {'attack_strength': 2.0}
        }
        
        config = ExperimentConfig.from_dict(config_dict)
        assert config.experiment_name == 'test_from_dict'
        assert config.dataset == DatasetType.MNIST
        assert config.federated.num_honest_clients == 8
        assert config.quantum.num_qubits == 15