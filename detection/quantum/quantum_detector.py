"""Quantum-enhanced Byzantine detection."""

import numpy as np
from typing import Dict, List, Optional
import logging

from core.interfaces import DetectionStrategy, ProjectionStrategy, ClientUpdate
from config.base_config import QuantumConfig
from .feature_maps import QuantumFeatureMapBuilder
from .embeddings import QuantumEmbeddingComputer

class QuantumByzantineDetector(DetectionStrategy):
    """Quantum-enhanced Byzantine detection using quantum feature maps"""
    
    def __init__(
        self,
        feature_map_type: str,
        quantum_config: QuantumConfig,
        projection_strategy: Optional[ProjectionStrategy] = None,
        projection_dim: int = 27
    ):
        self.feature_map_type = feature_map_type
        self.quantum_config = quantum_config
        self.projection_strategy = projection_strategy
        self.projection_dim = projection_dim
        self.logger = logging.getLogger(__name__)
        
        # Initialize quantum components
        self.feature_map_builder = QuantumFeatureMapBuilder(quantum_config)
        self.embedding_computer = QuantumEmbeddingComputer(quantum_config)
        
    def detect(
        self,
        updates: List[ClientUpdate],
        num_byzantine: int
    ) -> List[int]:
        """Perform quantum-enhanced Byzantine detection"""
        
        # Apply projection if configured
        if self.projection_strategy:
            projected_data = self.projection_strategy.project(updates, self.projection_dim)
        else:
            projected_data = self._flatten_updates(updates)
            
        # Compute quantum embeddings
        embeddings = self.embedding_computer.compute_embedding(
            projected_data,
            self.feature_map_type,
            self.quantum_config.num_qubits
        )
        
        if embeddings is None:
            self.logger.warning("Quantum embedding computation failed, falling back to classical")
            return self._classical_fallback(projected_data, num_byzantine)
        
        # Apply Multi-KRUM on quantum embeddings
        selected_indices = self._quantum_multikrum(embeddings, num_byzantine)
        
        return selected_indices
    
    def _flatten_updates(self, updates: List[ClientUpdate]) -> np.ndarray:
        """Flatten client updates for processing"""
        import torch
        flattened = []
        for update in updates:
            flat = torch.cat([w.view(-1) for w in update.weights])
            flattened.append(flat.cpu().numpy())
        return np.stack(flattened)
    
    def _quantum_multikrum(self, embedding_data: Dict[str, np.ndarray], num_byzantine: int) -> List[int]:
        """Apply Multi-KRUM on quantum embeddings"""
        if embedding_data is None:
            self.logger.error("Cannot perform Multi-KRUM selection: Embeddings not computed.")
            return []

        # Extract embeddings and basis states
        embeddings = embedding_data['embeddings']
        basis_states = embedding_data['basis_states']

        n = embeddings.shape[0]
        k = n - num_byzantine - 2
        if k <= 0:
            self.logger.warning(f"k={k} is too small in Multi-KRUM (n={n}, f={num_byzantine}). Selecting all clients.")
            return list(range(n))

        # Compute distances efficiently based on sparse representation
        scores = np.zeros(n)
        distances = np.zeros((n, n))

        # Compute pairwise distances
        for i in range(n):
            for j in range(i+1, n):
                # Compute distance between client i and j using sparse embeddings
                dist_squared = self._compute_sparse_distance(
                    embeddings[i], basis_states[i],
                    embeddings[j], basis_states[j]
                )
                
                # Store in distance matrix (symmetric)
                distances[i, j] = dist_squared
                distances[j, i] = dist_squared

        for i in range(n):
            # Find k closest neighbors (excluding self)
            neighbor_indices = np.argsort(distances[i, :])[1:k+1]
            scores[i] = np.sum(distances[i, neighbor_indices])

        num_to_select = n - num_byzantine
        selected_indices = np.argsort(scores)[:num_to_select]
        return sorted(list(selected_indices))
    
    def _compute_sparse_distance(
        self, 
        emb1: np.ndarray, states1: np.ndarray,
        emb2: np.ndarray, states2: np.ndarray
    ) -> float:
        """Compute distance between two sparse quantum embeddings"""
        # Find common basis states
        common_mask1 = np.zeros(len(states1), dtype=bool)
        common_mask2 = np.zeros(len(states2), dtype=bool)

        for idx1 in range(len(states1)):
            state1 = states1[idx1]
            if state1 == 0 and emb1[idx1] == 0:  # Skip empty entries
                continue

            for idx2 in range(len(states2)):
                state2 = states2[idx2]
                if state1 == state2:
                    common_mask1[idx1] = True
                    common_mask2[idx2] = True
                    break

        # Calculate squared distance
        dist_squared = 0.0

        # Add contribution of states in 1 not in 2
        dist_squared += np.sum(emb1[~common_mask1]**2)

        # Add contribution of states in 2 not in 1
        dist_squared += np.sum(emb2[~common_mask2]**2)

        # Add contribution of common states
        for idx1 in range(len(states1)):
            if common_mask1[idx1]:
                state1 = states1[idx1]
                # Find matching state in 2
                for idx2 in range(len(states2)):
                    if states2[idx2] == state1:
                        diff = emb1[idx1] - emb2[idx2]
                        dist_squared += diff**2
                        break

        return dist_squared
    
    def _classical_fallback(self, data: np.ndarray, num_byzantine: int) -> List[int]:
        """Fallback to classical Multi-KRUM if quantum computation fails"""
        from sklearn.metrics.pairwise import euclidean_distances
        
        n = data.shape[0]
        k = n - num_byzantine - 2
        
        if k <= 0:
            return list(range(n))
            
        distances = euclidean_distances(data, data) ** 2
        scores = np.zeros(n)
        
        for i in range(n):
            neighbor_indices = np.argsort(distances[i, :])[1:k+1]
            scores[i] = np.sum(distances[i, neighbor_indices])
            
        num_to_select = n - num_byzantine
        selected_indices = np.argsort(scores)[:num_to_select]
        
        return sorted(list(selected_indices))