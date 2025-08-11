"""Metrics calculation for Byzantine detection evaluation."""

from typing import Dict, List

class MetricsEvaluator:
    """Evaluates Byzantine detection performance metrics"""
    
    def calculate_metrics(
        self, 
        selected_indices: List[int], 
        num_honest: int, 
        num_byzantine: int
    ) -> Dict[str, float]:
        """Calculate standard Byzantine detection metrics"""
        
        n = num_honest + num_byzantine
        honest_indices = set(range(num_honest))
        byzantine_indices = set(range(num_honest, n))
        
        # Ensure selected indices are valid
        valid_selected_indices = []
        for idx in selected_indices:
            try:
                int_idx = int(idx)
                if 0 <= int_idx < n:
                    valid_selected_indices.append(int_idx)
            except (ValueError, TypeError):
                pass
        
        selected_set = set(valid_selected_indices)
        rejected_set = set(range(n)) - selected_set
        
        # Calculate confusion matrix elements
        tp = len(rejected_set.intersection(byzantine_indices))  # Correctly rejected Byzantine
        fp = len(rejected_set.intersection(honest_indices))     # Incorrectly rejected Honest
        tn = len(selected_set.intersection(honest_indices))     # Correctly selected Honest
        fn = len(selected_set.intersection(byzantine_indices))  # Incorrectly selected Byzantine
        
        # Calculate derived metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Byzantine rejection rate
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        honest_retention = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return {
            'TP': tp,
            'FP': fp, 
            'TN': tn,
            'FN': fn,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'HonestRetention': honest_retention
        }