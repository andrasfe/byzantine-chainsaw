"""Dataset loading utilities."""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import Tuple

from config.base_config import DatasetType

class DatasetLoader:
    """Utility class for loading different datasets"""
    
    def __init__(self, root_dir: str = './data'):
        self.root_dir = root_dir
    
    def load_cifar10(self) -> Tuple[torchvision.datasets.CIFAR10, DataLoader]:
        """Load CIFAR-10 dataset"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        trainset = torchvision.datasets.CIFAR10(
            root=self.root_dir, 
            train=True, 
            download=True, 
            transform=transform
        )
        
        testset = torchvision.datasets.CIFAR10(
            root=self.root_dir, 
            train=False, 
            download=True, 
            transform=transform
        )
        
        test_loader = DataLoader(testset, batch_size=128, shuffle=False)
        
        print(f"Loaded CIFAR-10: {len(trainset)} train, {len(testset)} test")
        return trainset, test_loader
    
    def load_mnist(self) -> Tuple[torchvision.datasets.MNIST, DataLoader]:
        """Load MNIST dataset"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        trainset = torchvision.datasets.MNIST(
            root=self.root_dir,
            train=True,
            download=True,
            transform=transform
        )
        
        testset = torchvision.datasets.MNIST(
            root=self.root_dir,
            train=False,
            download=True,
            transform=transform
        )
        
        test_loader = DataLoader(testset, batch_size=128, shuffle=False)
        
        print(f"Loaded MNIST: {len(trainset)} train, {len(testset)} test")
        return trainset, test_loader
    
    def load_dataset(self, dataset_type: DatasetType) -> Tuple:
        """Load dataset based on type"""
        if dataset_type == DatasetType.CIFAR10:
            return self.load_cifar10()
        elif dataset_type == DatasetType.MNIST:
            return self.load_mnist()
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")