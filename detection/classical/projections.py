"""Projection strategies for dimensionality reduction."""

import torch
import numpy as np
from typing import List

from core.interfaces import ProjectionStrategy, ClientUpdate

class RandomProjection(ProjectionStrategy):
    """Random Gaussian projection for dimensionality reduction"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        
    def project(
        self, 
        updates: List[ClientUpdate],
        target_dim: int
    ) -> np.ndarray:
        """Project updates using random Gaussian matrix"""
        # Flatten updates
        flattened_updates = []
        for client_upd in updates:
            flat_upd = torch.cat([p.view(-1) for p in client_upd.weights])
            flattened_updates.append(flat_upd)

        update_matrix = torch.stack(flattened_updates).to(self.device)
        original_dim = update_matrix.shape[1]

        # Create random projection matrix
        projection_matrix = torch.randn(
            original_dim, target_dim, device=self.device
        ) / np.sqrt(target_dim)
        
        # Project the updates
        projected_updates = torch.matmul(update_matrix, projection_matrix)

        return projected_updates.cpu().numpy()

class ImportanceWeightedProjection(ProjectionStrategy):
    """Importance-weighted projection based on parameter variance"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        
    def project(
        self,
        updates: List[ClientUpdate],
        target_dim: int
    ) -> np.ndarray:
        """Project updates using importance-weighted random projection"""
        # Flatten updates
        flattened_updates = []
        for client_upd in updates:
            flat_upd = torch.cat([p.view(-1) for p in client_upd.weights])
            flattened_updates.append(flat_upd)

        update_matrix = torch.stack(flattened_updates).to(self.device)
        original_dim = update_matrix.shape[1]

        # Calculate importance weights based on variance across clients
        variance = torch.var(update_matrix, dim=0)
        importance_weights = variance / (torch.sum(variance) + 1e-8)

        # Create weighted projection matrix
        projection_matrix = torch.randn(original_dim, target_dim, device=self.device)
        
        # Apply importance weights
        for i in range(original_dim):
            projection_matrix[i, :] *= torch.sqrt(importance_weights[i])

        # Normalize projection matrix columns
        for j in range(target_dim):
            col_norm = torch.norm(projection_matrix[:, j])
            if col_norm > 0:
                projection_matrix[:, j] /= col_norm

        # Project the updates
        projected_updates = torch.matmul(update_matrix, projection_matrix)

        return projected_updates.cpu().numpy()