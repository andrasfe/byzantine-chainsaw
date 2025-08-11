"""Tests for metrics evaluation."""

import pytest
import numpy as np

from metrics.evaluator import MetricsEvaluator

class TestMetricsEvaluator:
    """Test metrics evaluation functionality"""
    
    def test_perfect_detection(self):
        """Test metrics with perfect Byzantine detection"""
        evaluator = MetricsEvaluator()
        
        # Perfect case: all honest selected, all Byzantine rejected
        # 5 honest (indices 0-4), 2 Byzantine (indices 5-6)
        # Select all honest clients (0-4)
        selected_indices = [0, 1, 2, 3, 4]
        
        metrics = evaluator.calculate_metrics(
            selected_indices=selected_indices,
            num_honest=5,
            num_byzantine=2
        )
        
        assert metrics['TP'] == 2  # Both Byzantine correctly rejected
        assert metrics['FP'] == 0  # No honest incorrectly rejected
        assert metrics['TN'] == 5  # All honest correctly selected
        assert metrics['FN'] == 0  # No Byzantine incorrectly selected
        assert metrics['Precision'] == 1.0
        assert metrics['Recall'] == 1.0  # Byzantine rejection rate
        assert metrics['F1'] == 1.0
        assert metrics['HonestRetention'] == 1.0
        
    def test_poor_detection(self):
        """Test metrics with poor Byzantine detection"""
        evaluator = MetricsEvaluator()
        
        # Poor case: some honest rejected, some Byzantine selected
        # 4 honest (indices 0-3), 3 Byzantine (indices 4-6)
        # Select mix: 2 honest + 1 Byzantine
        selected_indices = [0, 1, 4]  # 2 honest + 1 Byzantine
        
        metrics = evaluator.calculate_metrics(
            selected_indices=selected_indices,
            num_honest=4,
            num_byzantine=3
        )
        
        assert metrics['TP'] == 2  # 2 Byzantine correctly rejected (indices 5,6)
        assert metrics['FP'] == 2  # 2 honest incorrectly rejected (indices 2,3)
        assert metrics['TN'] == 2  # 2 honest correctly selected (indices 0,1)
        assert metrics['FN'] == 1  # 1 Byzantine incorrectly selected (index 4)
        
        # Check calculated metrics
        assert metrics['Precision'] == 2/4  # TP/(TP+FP) = 2/4 = 0.5
        assert metrics['Recall'] == 2/3     # TP/(TP+FN) = 2/3 ≈ 0.667
        assert abs(metrics['F1'] - (2 * 0.5 * (2/3) / (0.5 + 2/3))) < 1e-6
        assert metrics['HonestRetention'] == 2/4  # TN/(TN+FP) = 2/4 = 0.5
        
    def test_no_byzantine_detected(self):
        """Test case where no Byzantine clients are detected"""
        evaluator = MetricsEvaluator()
        
        # Select all clients (including Byzantine)
        # 3 honest (0-2), 2 Byzantine (3-4)
        selected_indices = [0, 1, 2, 3, 4]
        
        metrics = evaluator.calculate_metrics(
            selected_indices=selected_indices,
            num_honest=3,
            num_byzantine=2
        )
        
        assert metrics['TP'] == 0  # No Byzantine rejected
        assert metrics['FP'] == 0  # No honest rejected
        assert metrics['TN'] == 3  # All honest selected
        assert metrics['FN'] == 2  # All Byzantine selected
        assert metrics['Precision'] == 0.0  # No true positives
        assert metrics['Recall'] == 0.0     # No Byzantine detected
        assert metrics['F1'] == 0.0
        assert metrics['HonestRetention'] == 1.0  # All honest retained
        
    def test_all_byzantine_detected(self):
        """Test case where all clients are rejected"""
        evaluator = MetricsEvaluator()
        
        # Reject all clients
        # 2 honest (0-1), 2 Byzantine (2-3)
        selected_indices = []  # Nobody selected
        
        metrics = evaluator.calculate_metrics(
            selected_indices=selected_indices,
            num_honest=2,
            num_byzantine=2
        )
        
        assert metrics['TP'] == 2  # All Byzantine rejected
        assert metrics['FP'] == 2  # All honest rejected
        assert metrics['TN'] == 0  # No honest selected
        assert metrics['FN'] == 0  # No Byzantine selected
        assert metrics['Precision'] == 0.5  # TP/(TP+FP) = 2/4
        assert metrics['Recall'] == 1.0     # All Byzantine detected
        assert abs(metrics['F1'] - (2 * 0.5 * 1.0 / (0.5 + 1.0))) < 1e-6
        assert metrics['HonestRetention'] == 0.0  # No honest retained
        
    def test_invalid_indices_handling(self):
        """Test handling of invalid selected indices"""
        evaluator = MetricsEvaluator()
        
        # Include invalid indices (out of range, non-numeric)
        # Note: 2.5 gets converted to int(2.5) = 2, which is valid
        selected_indices = [0, 1, 10, -1, 'invalid', 7.5]  # 7.5 -> 7 which is out of range
        
        metrics = evaluator.calculate_metrics(
            selected_indices=selected_indices,
            num_honest=3,
            num_byzantine=2
        )
        
        # Should only consider valid indices [0, 1] (7.5->7, 10, -1, 'invalid' are filtered out)
        # With 3 honest (0-2) and 2 Byzantine (3-4), selecting [0,1] means:
        # - Selected: 0,1 (2 honest, 0 Byzantine)
        # - Rejected: 2,3,4 (1 honest, 2 Byzantine)
        assert metrics['TP'] == 2   # Both Byzantine rejected (indices 3,4)
        assert metrics['FP'] == 1   # One honest rejected (index 2)
        assert metrics['TN'] == 2   # Two honest selected (indices 0,1)
        assert metrics['FN'] == 0   # No Byzantine selected
        
    def test_edge_case_division_by_zero(self):
        """Test edge cases that could cause division by zero"""
        evaluator = MetricsEvaluator()
        
        # Case where TP + FP = 0 (nobody rejected)
        selected_indices = [0, 1, 2]  # All selected
        
        metrics = evaluator.calculate_metrics(
            selected_indices=selected_indices,
            num_honest=2,
            num_byzantine=1
        )
        
        # Should handle division by zero gracefully
        assert metrics['Precision'] == 0.0
        assert isinstance(metrics['F1'], float)
        
    def test_numpy_indices(self):
        """Test with numpy array indices"""
        evaluator = MetricsEvaluator()
        
        # Use numpy array for selected indices
        selected_indices = np.array([0, 2, 3])
        
        metrics = evaluator.calculate_metrics(
            selected_indices=selected_indices,
            num_honest=3,
            num_byzantine=2
        )
        
        # Should work with numpy arrays
        assert isinstance(metrics, dict)
        assert 'TP' in metrics