"""Core interfaces and abstract base classes for the quantum Byzantine detection system."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import torch
from dataclasses import dataclass
import numpy as np

@dataclass
class ClientUpdate:
    """Encapsulates a client's model update"""
    client_id: int
    weights: List[torch.Tensor]
    metadata: Dict[str, Any]
    is_byzantine: bool = False

class ModelInterface(ABC):
    """Abstract interface for ML models"""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
        pass
    
    @abstractmethod
    def get_weights(self) -> List[torch.Tensor]:
        """Get model weights as a list of tensors"""
        pass
    
    @abstractmethod
    def set_weights(self, weights: List[torch.Tensor]) -> None:
        """Set model weights from a list of tensors"""
        pass
    
    @abstractmethod
    def train_step(self, data: torch.Tensor, labels: torch.Tensor) -> float:
        """Perform one training step and return loss"""
        pass

class AttackStrategy(ABC):
    """Abstract interface for Byzantine attacks"""
    
    @abstractmethod
    def generate_byzantine_update(
        self, 
        honest_updates: List[ClientUpdate],
        global_weights: List[torch.Tensor]
    ) -> ClientUpdate:
        """Generate a Byzantine update based on honest updates"""
        pass

class DetectionStrategy(ABC):
    """Abstract interface for Byzantine detection"""
    
    @abstractmethod
    def detect(
        self, 
        updates: List[ClientUpdate],
        num_byzantine: int
    ) -> List[int]:
        """Returns indices of selected (non-Byzantine) clients"""
        pass

class ProjectionStrategy(ABC):
    """Abstract interface for dimensionality reduction"""
    
    @abstractmethod
    def project(
        self, 
        updates: List[ClientUpdate],
        target_dim: int
    ) -> np.ndarray:
        """Project updates to lower dimensional space"""
        pass

class AggregationStrategy(ABC):
    """Abstract interface for update aggregation"""
    
    @abstractmethod
    def aggregate(
        self,
        updates: List[ClientUpdate],
        selected_indices: List[int]
    ) -> List[torch.Tensor]:
        """Aggregate selected updates into global update"""
        pass

class QuantumEmbedding(ABC):
    """Abstract interface for quantum embeddings"""
    
    @abstractmethod
    def compute_embedding(
        self,
        data: np.ndarray,
        num_qubits: int
    ) -> Dict[str, np.ndarray]:
        """Compute quantum embedding for input data"""
        pass