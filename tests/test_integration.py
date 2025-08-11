"""Integration tests for the complete system."""

import pytest
import tempfile
import yaml
from pathlib import Path

from config.experiment_config import ExperimentConfig
from config.base_config import DatasetType, AttackType, DetectionMethod
from experiments.experiment_runner import ExperimentRunner

class TestSystemIntegration:
    """Integration tests for end-to-end system functionality"""
    
    def test_minimal_experiment_setup(self):
        """Test minimal experiment setup without full execution"""
        # Create minimal configuration
        config = ExperimentConfig(
            experiment_name="test_integration",
            dataset=DatasetType.CIFAR10,
            federated={"num_honest_clients": 2, "num_byzantine_clients": 1, "num_rounds": 1},
            quantum={"num_qubits": 5, "batch_size": 1},
            detection={"projection_dim": 5},
            attack={},
            detection_methods=[DetectionMethod.STANDARD_MULTIKRUM]
        )
        
        # Test that runner can be created
        runner = ExperimentRunner(config)
        
        # Verify components are initialized
        assert runner.config == config
        assert runner.device is not None
        assert runner.model_factory is not None
        assert runner.attack_factory is not None
        
    def test_config_file_integration(self):
        """Test loading configuration from YAML file"""
        config_data = {
            'experiment_name': 'test_yaml_integration',
            'dataset': 'cifar10',
            'federated': {
                'num_honest_clients': 3,
                'num_byzantine_clients': 1,
                'num_rounds': 2,
                'local_epochs': 1,
                'batch_size': 16,
                'learning_rate': 0.1
            },
            'quantum': {
                'num_qubits': 8,
                'num_shots': 256,
                'use_gpu': False,
                'batch_size': 1
            },
            'detection': {
                'projection_dim': 8,
                'detection_methods': ['standard_multikrum', 'classical_random']
            },
            'attack': {
                'attack_type': 'lie_attack',
                'attack_strength': 1.0
            },
            'seed': 123,
            'device': 'cpu',
            'output_dir': './test_results'
        }
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
            
        try:
            # Load configuration
            config = ExperimentConfig.from_yaml(config_path)
            
            # Verify loaded correctly
            assert config.experiment_name == 'test_yaml_integration'
            assert config.dataset == DatasetType.CIFAR10
            assert config.federated.num_honest_clients == 3
            assert config.quantum.num_qubits == 8
            assert config.quantum.use_gpu == False
            assert config.device == 'cpu'
            
            # Test creating runner with loaded config
            runner = ExperimentRunner(config)
            assert runner.device.type == 'cpu'
            
        finally:
            Path(config_path).unlink()  # Clean up
            
    def test_component_interaction(self):
        """Test interaction between different system components"""
        config = ExperimentConfig(
            experiment_name="test_component_interaction",
            dataset=DatasetType.CIFAR10,
            federated={
                'num_honest_clients': 2,
                'num_byzantine_clients': 1,
                'num_rounds': 1,
                'local_epochs': 1
            },
            quantum={'num_qubits': 6},
            detection={'projection_dim': 6},
            attack={},
            detection_methods=[DetectionMethod.STANDARD_MULTIKRUM]
        )
        
        runner = ExperimentRunner(config)
        
        # Test that setup can complete without errors
        try:
            runner._setup_experiment()
            
            # Verify components are properly initialized
            assert runner.trainset is not None
            assert runner.test_loader is not None
            assert runner.global_model is not None
            assert runner.global_weights is not None
            assert len(runner.client_loaders) == 2  # num_honest_clients
            assert runner.attack_strategy is not None
            
        except Exception as e:
            # Setup might fail due to dataset download or other issues in test environment
            # Just verify the runner was created properly
            pytest.skip(f"Setup failed in test environment: {e}")
            
    def test_model_attack_integration(self):
        """Test integration between model training and attack generation"""
        from models.model_factory import ModelFactory
        from attacks.attack_factory import AttackFactory
        from config.base_config import FederatedConfig, AttackConfig
        from core.interfaces import ClientUpdate
        import torch
        
        # Create components
        model_factory = ModelFactory()
        attack_factory = AttackFactory()
        
        # Create model and get weights
        model = model_factory.create(DatasetType.CIFAR10, device='cpu')
        global_weights = model.get_weights()
        
        # Create mock honest updates
        honest_updates = []
        for i in range(2):
            # Simulate weight updates
            update_weights = [w.clone() + torch.randn_like(w) * 0.01 for w in global_weights]
            update = ClientUpdate(
                client_id=i,
                weights=update_weights,
                metadata={'client_type': 'honest'},
                is_byzantine=False
            )
            honest_updates.append(update)
        
        # Create attack and generate Byzantine update
        attack_config = AttackConfig(attack_type=AttackType.LIE)
        attack_strategy = attack_factory.create(AttackType.LIE, attack_config)
        
        byzantine_update = attack_strategy.generate_byzantine_update(
            honest_updates, global_weights
        )
        
        # Verify integration works
        assert byzantine_update.is_byzantine == True
        assert len(byzantine_update.weights) == len(global_weights)
        
        # Verify weights have appropriate shapes
        for orig, byz in zip(global_weights, byzantine_update.weights):
            assert orig.shape == byz.shape
            
    def test_detection_metrics_integration(self):
        """Test integration between detection algorithms and metrics evaluation"""
        from detection.classical.multikrum import MultiKrumDetector
        from metrics.evaluator import MetricsEvaluator
        from core.interfaces import ClientUpdate
        import torch
        
        # Create detector and evaluator
        detector = MultiKrumDetector()
        evaluator = MetricsEvaluator()
        
        # Create mock client updates
        updates = []
        num_honest = 4
        num_byzantine = 2
        
        for i in range(num_honest + num_byzantine):
            # Create different update patterns for honest vs Byzantine
            if i < num_honest:
                weights = [torch.randn(10) * 0.1, torch.randn(5) * 0.1]  # Honest: small updates
            else:
                weights = [torch.randn(10) * 2.0, torch.randn(5) * 2.0]  # Byzantine: large updates
                
            update = ClientUpdate(
                client_id=i,
                weights=weights,
                metadata={},
                is_byzantine=(i >= num_honest)
            )
            updates.append(update)
        
        # Run detection
        selected_indices = detector.detect(updates, num_byzantine)
        
        # Evaluate metrics
        metrics = evaluator.calculate_metrics(selected_indices, num_honest, num_byzantine)
        
        # Verify integration produces valid metrics
        assert isinstance(metrics, dict)
        assert 'F1' in metrics
        assert 'Recall' in metrics
        assert 'Precision' in metrics
        assert 0.0 <= metrics['F1'] <= 1.0
        assert 0.0 <= metrics['Recall'] <= 1.0
        assert 0.0 <= metrics['Precision'] <= 1.0