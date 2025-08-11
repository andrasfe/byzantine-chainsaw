"""Federated learning data management."""

import random
from typing import List
from torch.utils.data import DataLoader, Subset
import torch

class FederatedDataManager:
    """Manages data partitioning for federated learning"""
    
    def __init__(self, dataset, batch_size: int = 32):
        self.dataset = dataset
        self.batch_size = batch_size
        
    def partition_data(self, num_clients: int) -> List[List[int]]:
        """Partition dataset among clients (IID distribution)"""
        samples_per_client = len(self.dataset) // num_clients
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        
        client_data_indices = []
        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client
            client_data_indices.append(indices[start_idx:end_idx])
        
        return client_data_indices
    
    def create_client_loaders(self, client_indices: List[List[int]]) -> List[DataLoader]:
        """Create data loaders for each client"""
        client_loaders = []
        for indices in client_indices:
            subset = Subset(self.dataset, indices)
            loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True)
            client_loaders.append(loader)
        
        return client_loaders