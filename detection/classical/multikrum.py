"""Multi-KRUM Byzantine detection algorithm."""

import torch
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from typing import List, Optional

from core.interfaces import DetectionStrategy, ProjectionStrategy, ClientUpdate

class MultiKrumDetector(DetectionStrategy):
    """Multi-KRUM Byzantine detection algorithm"""
    
    def __init__(
        self,
        projection_strategy: Optional[ProjectionStrategy] = None,
        detection_noise: float = 0.0,
        projection_dim: int = 27
    ):
        self.projection_strategy = projection_strategy
        self.detection_noise = detection_noise
        self.projection_dim = projection_dim
        
    def detect(
        self,
        updates: List[ClientUpdate],
        num_byzantine: int
    ) -> List[int]:
        """Detect Byzantine clients using Multi-KRUM algorithm"""
        
        # Apply projection if configured
        if self.projection_strategy:
            data = self.projection_strategy.project(updates, self.projection_dim)
        else:
            data = self._flatten_updates(updates)
            
        return self._perform_detection(data, num_byzantine)
    
    def _flatten_updates(self, updates: List[ClientUpdate]) -> np.ndarray:
        """Flatten client updates for processing"""
        flattened = []
        for update in updates:
            flat = torch.cat([w.view(-1) for w in update.weights])
            flattened.append(flat.cpu().numpy())
        return np.stack(flattened)
    
    def _perform_detection(
        self,
        data: np.ndarray,
        num_byzantine: int
    ) -> List[int]:
        """Perform Multi-KRUM detection on flattened data"""
        n = data.shape[0]
        k = n - num_byzantine - 2
        
        if k <= 0:
            print(f"Warning: k={k} is too small in Multi-KRUM (n={n}, f={num_byzantine}). Selecting all clients.")
            return list(range(n))
            
        # Add detection noise if configured
        if self.detection_noise > 0:
            noise = np.random.normal(
                0, 
                self.detection_noise * np.std(data),
                data.shape
            )
            data = data + noise
            
        # Calculate distances and scores
        distances = euclidean_distances(data, data) ** 2
        scores = np.zeros(n)
        
        for i in range(n):
            # Find k closest neighbors (excluding self)
            neighbor_indices = np.argsort(distances[i, :])[1:k+1]
            scores[i] = np.sum(distances[i, neighbor_indices])
            
        # Select clients with lowest scores
        num_to_select = n - num_byzantine
        selected_indices = np.argsort(scores)[:num_to_select]
        
        return sorted(list(selected_indices))