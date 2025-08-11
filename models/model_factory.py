"""Factory for creating ML models."""

from typing import Dict, Any
import torch

from core.interfaces import ModelInterface
from .cnn_models import SimpleCNN
from config.base_config import DatasetType

class ModelFactory:
    """Factory for creating ML models based on configuration"""
    
    def __init__(self):
        self._models = {
            DatasetType.CIFAR10: SimpleCNN,
            DatasetType.CIFAR100: lambda: SimpleCNN(num_classes=100),
            DatasetType.MNIST: lambda: SimpleCNN(num_classes=10)  # Note: Would need architecture changes for MNIST
        }
    
    def create(self, dataset_type: DatasetType, device: str = "cpu", **kwargs) -> ModelInterface:
        """Create a model for the specified dataset"""
        if dataset_type not in self._models:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        model_class = self._models[dataset_type]
        
        # Handle lambda functions
        if callable(model_class) and not isinstance(model_class, type):
            model = model_class()
        else:
            model = model_class(**kwargs)
        
        # Move to specified device
        if device != "cpu":
            model = model.to(device)
        
        return model
    
    def register_model(self, dataset_type: DatasetType, model_class):
        """Register a new model class for a dataset type"""
        self._models[dataset_type] = model_class