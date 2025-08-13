#!/usr/bin/env python3
"""Quick test to verify the classical embeddings work on projected matrices"""

import numpy as np
import sys
sys.path.append('.')
from qe_multikrum import polynomial_feature_map_from_matrix, rbf_feature_map_from_matrix, fourier_feature_map_from_matrix

def test_matrix_functions():
    print("Testing classical embeddings on projected matrices...")
    
    # Create mock projected data (18 clients, 27 dimensions)
    np.random.seed(42)
    projected_matrix = np.random.randn(18, 27)
    
    print(f"Input matrix shape: {projected_matrix.shape}")
    
    # Test polynomial features
    try:
        poly_out = polynomial_feature_map_from_matrix(projected_matrix, projection_dim=27)
        print(f"✓ Polynomial features: {poly_out.shape}")
    except Exception as e:
        print(f"✗ Polynomial failed: {e}")
    
    # Test RBF features  
    try:
        rbf_out = rbf_feature_map_from_matrix(projected_matrix, n_components=27)
        print(f"✓ RBF features: {rbf_out.shape}")
    except Exception as e:
        print(f"✗ RBF failed: {e}")
    
    # Test Fourier features
    try:
        fourier_out = fourier_feature_map_from_matrix(projected_matrix, n_components=27)
        print(f"✓ Fourier features: {fourier_out.shape}")
    except Exception as e:
        print(f"✗ Fourier failed: {e}")
    
    print("Test complete!")

if __name__ == "__main__":
    test_matrix_functions()