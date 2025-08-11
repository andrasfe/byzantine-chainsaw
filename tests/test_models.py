"""Tests for neural network models."""

import pytest
import torch

from models.cnn_models import SimpleCNN
from models.model_factory import ModelFactory
from config.base_config import DatasetType

class TestSimpleCNN:
    """Test SimpleCNN model implementation"""
    
    def test_model_creation(self):
        """Test creating SimpleCNN model"""
        model = SimpleCNN(num_classes=10)
        assert isinstance(model, SimpleCNN)
        
    def test_forward_pass(self):
        """Test forward pass with CIFAR-10 input"""
        model = SimpleCNN(num_classes=10)
        # CIFAR-10 input: batch_size=2, channels=3, height=32, width=32
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert output.shape == (2, 10)
        
    def test_get_set_weights(self):
        """Test getting and setting model weights"""
        model = SimpleCNN(num_classes=10)
        
        # Get original weights
        original_weights = model.get_weights()
        assert isinstance(original_weights, list)
        assert len(original_weights) > 0
        
        # Modify weights
        modified_weights = [w.clone() + 0.1 for w in original_weights]
        model.set_weights(modified_weights)
        
        # Verify weights changed
        new_weights = model.get_weights()
        for orig, new in zip(original_weights, new_weights):
            assert not torch.equal(orig, new)
            
    def test_train_step(self):
        """Test single training step"""
        model = SimpleCNN(num_classes=10)
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 10, (4,))
        
        loss = model.train_step(x, y)
        assert isinstance(loss, float)
        assert loss > 0  # Cross-entropy loss should be positive
        
    def test_evaluate(self):
        """Test model evaluation"""
        model = SimpleCNN(num_classes=10)
        
        # Create simple test loader
        from torch.utils.data import DataLoader, TensorDataset
        x = torch.randn(20, 3, 32, 32)
        y = torch.randint(0, 10, (20,))
        dataset = TensorDataset(x, y)
        test_loader = DataLoader(dataset, batch_size=4)
        
        accuracy = model.evaluate(test_loader, "cpu")
        assert 0.0 <= accuracy <= 100.0

class TestModelFactory:
    """Test model factory implementation"""
    
    def test_create_cifar10_model(self):
        """Test creating CIFAR-10 model"""
        factory = ModelFactory()
        model = factory.create(DatasetType.CIFAR10)
        assert isinstance(model, SimpleCNN)
        
    def test_create_with_device(self):
        """Test creating model with specific device"""
        factory = ModelFactory()
        model = factory.create(DatasetType.CIFAR10, device="cpu")
        
        # Check that model parameters are on CPU
        for param in model.parameters():
            assert param.device.type == "cpu"
            
    def test_unsupported_dataset(self):
        """Test error handling for unsupported dataset"""
        factory = ModelFactory()
        
        # This should work with current implementation
        model = factory.create(DatasetType.MNIST)
        assert isinstance(model, SimpleCNN)
        
    def test_register_new_model(self):
        """Test registering new model type"""
        factory = ModelFactory()
        
        # Define dummy model class
        class DummyModel(SimpleCNN):
            pass
            
        # Register and test
        factory.register_model(DatasetType.CIFAR100, DummyModel)
        model = factory.create(DatasetType.CIFAR100)
        assert isinstance(model, DummyModel)