"""Tests for Byzantine detection algorithms."""

import pytest
import torch
import numpy as np

from detection.classical.multikrum import MultiKrumDetector
from detection.classical.projections import RandomProjection, ImportanceWeightedProjection
from core.interfaces import ClientUpdate
from config.base_config import QuantumConfig

class TestMultiKrumDetector:
    """Test Multi-KRUM detection algorithm"""
    
    def test_detector_creation(self):
        """Test creating MultiKrumDetector"""
        detector = MultiKrumDetector()
        assert detector.projection_strategy is None
        assert detector.detection_noise == 0.0
        
    def test_detection_without_projection(self):
        """Test detection without projection"""
        detector = MultiKrumDetector()
        
        # Create mock client updates
        updates = []
        for i in range(5):
            # Create diverse updates for proper clustering
            if i < 3:  # Honest clients - similar updates
                weights = [torch.randn(10) * 0.1, torch.randn(5) * 0.1]
            else:  # Byzantine clients - different updates  
                weights = [torch.randn(10) * 2.0, torch.randn(5) * 2.0]
                
            update = ClientUpdate(
                client_id=i,
                weights=weights,
                metadata={},
                is_byzantine=(i >= 3)
            )
            updates.append(update)
            
        # Detect with 2 Byzantine clients
        selected_indices = detector.detect(updates, num_byzantine=2)
        
        # Should select 3 clients (5 - 2 = 3)
        assert len(selected_indices) == 3
        assert all(idx in range(5) for idx in selected_indices)
        
    def test_detection_with_projection(self):
        """Test detection with random projection"""
        projection = RandomProjection()
        detector = MultiKrumDetector(
            projection_strategy=projection,
            projection_dim=8
        )
        
        # Create mock updates
        updates = []
        for i in range(4):
            weights = [torch.randn(20), torch.randn(10)]
            update = ClientUpdate(
                client_id=i,
                weights=weights,
                metadata={},
                is_byzantine=False
            )
            updates.append(update)
            
        selected_indices = detector.detect(updates, num_byzantine=1)
        
        # Should select 3 clients (4 - 1 = 3)
        assert len(selected_indices) == 3
        
    def test_edge_case_too_few_clients(self):
        """Test edge case with too few clients"""
        detector = MultiKrumDetector()
        
        # Only 2 clients, but 2 Byzantine - impossible scenario
        updates = []
        for i in range(2):
            weights = [torch.randn(5)]
            update = ClientUpdate(client_id=i, weights=weights, metadata={}, is_byzantine=False)
            updates.append(update)
            
        selected_indices = detector.detect(updates, num_byzantine=2)
        
        # Should select all clients when k <= 0
        assert len(selected_indices) == 2
        assert selected_indices == [0, 1]
        
    def test_detection_noise(self):
        """Test detection with added noise"""
        detector = MultiKrumDetector(detection_noise=0.1)
        
        updates = []
        for i in range(4):
            weights = [torch.randn(10)]
            update = ClientUpdate(client_id=i, weights=weights, metadata={}, is_byzantine=False)
            updates.append(update)
            
        selected_indices = detector.detect(updates, num_byzantine=1)
        assert len(selected_indices) == 3

class TestProjectionStrategies:
    """Test projection strategies"""
    
    def test_random_projection(self):
        """Test random projection"""
        projection = RandomProjection()
        
        # Create mock updates
        updates = []
        for i in range(3):
            weights = [torch.randn(20), torch.randn(15)]  # Total dim = 35
            update = ClientUpdate(client_id=i, weights=weights, metadata={}, is_byzantine=False)
            updates.append(update)
            
        projected = projection.project(updates, target_dim=10)
        
        assert projected.shape == (3, 10)
        assert isinstance(projected, np.ndarray)
        
    def test_importance_weighted_projection(self):
        """Test importance-weighted projection"""
        projection = ImportanceWeightedProjection()
        
        # Create mock updates with different variance patterns
        updates = []
        for i in range(4):
            # Create weights with varying patterns
            w1 = torch.randn(10) * (i + 1)  # Different scales
            w2 = torch.randn(5) * 0.1
            weights = [w1, w2]
            
            update = ClientUpdate(client_id=i, weights=weights, metadata={}, is_byzantine=False)
            updates.append(update)
            
        projected = projection.project(updates, target_dim=8)
        
        assert projected.shape == (4, 8)
        assert isinstance(projected, np.ndarray)
        
    def test_projection_device_handling(self):
        """Test projection with different devices"""
        projection = RandomProjection(device="cpu")
        
        updates = []
        for i in range(2):
            weights = [torch.randn(15), torch.randn(10)]
            update = ClientUpdate(client_id=i, weights=weights, metadata={}, is_byzantine=False)
            updates.append(update)
            
        projected = projection.project(updates, target_dim=12)
        
        assert projected.shape == (2, 12)
        # Should work with CPU device specification