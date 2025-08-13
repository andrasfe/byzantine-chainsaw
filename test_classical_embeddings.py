#!/usr/bin/env python3
"""
Unit tests for classical non-linear embeddings to verify:
1. Memory usage is reasonable
2. They work correctly on projected weights (not raw updates)
3. Output dimensions are correct
"""

import numpy as np
import torch
import time
import psutil
import os
from qe_multikrum import polynomial_feature_map, rbf_feature_map, fourier_feature_map, random_projection

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def create_mock_updates(num_clients=18, param_size=100000):
    """Create mock client updates similar to CNN gradients"""
    updates = []
    for i in range(num_clients):
        # Simulate flattened CNN gradients 
        update = [
            torch.randn(param_size // 4),  # conv1 weights
            torch.randn(param_size // 4),  # conv2 weights  
            torch.randn(param_size // 4),  # fc1 weights
            torch.randn(param_size // 4),  # fc2 weights
        ]
        updates.append(update)
    return updates

def test_memory_usage():
    """Test memory usage of classical embeddings"""
    print("=== Memory Usage Test ===")
    
    # Test with different scales
    test_configs = [
        (18, 10000, "Small"),
        (18, 50000, "Medium"), 
        (18, 100000, "Large")
    ]
    
    for num_clients, param_size, scale in test_configs:
        print(f"\n{scale} test: {num_clients} clients, {param_size} parameters each")
        
        # Create mock data
        updates = create_mock_updates(num_clients, param_size)
        
        initial_memory = get_memory_usage()
        print(f"Initial memory: {initial_memory:.1f} MB")
        
        # Test polynomial features
        start_time = time.time()
        start_memory = get_memory_usage()
        
        try:
            poly_features = polynomial_feature_map(updates, projection_dim=27)
            poly_time = time.time() - start_time
            poly_memory = get_memory_usage() - start_memory
            print(f"Polynomial: {poly_time:.2f}s, +{poly_memory:.1f}MB, shape: {poly_features.shape}")
            del poly_features
        except Exception as e:
            print(f"Polynomial FAILED: {e}")
        
        # Test RBF features
        start_time = time.time()
        start_memory = get_memory_usage()
        
        try:
            rbf_features = rbf_feature_map(updates, n_components=27)
            rbf_time = time.time() - start_time
            rbf_memory = get_memory_usage() - start_memory
            print(f"RBF: {rbf_time:.2f}s, +{rbf_memory:.1f}MB, shape: {rbf_features.shape}")
            del rbf_features
        except Exception as e:
            print(f"RBF FAILED: {e}")
        
        # Test Fourier features
        start_time = time.time()
        start_memory = get_memory_usage()
        
        try:
            fourier_features = fourier_feature_map(updates, n_components=27)
            fourier_time = time.time() - start_time
            fourier_memory = get_memory_usage() - start_memory
            print(f"Fourier: {fourier_time:.2f}s, +{fourier_memory:.1f}MB, shape: {fourier_features.shape}")
            del fourier_features
        except Exception as e:
            print(f"Fourier FAILED: {e}")

def test_projected_vs_raw():
    """Test that embeddings work on both projected and raw updates"""
    print("\n=== Projected vs Raw Updates Test ===")
    
    # Create mock updates
    updates = create_mock_updates(18, 50000)
    
    # Get raw embedding
    print("Computing on RAW updates...")
    raw_poly = polynomial_feature_map(updates, projection_dim=27)
    raw_rbf = rbf_feature_map(updates, n_components=27)
    raw_fourier = fourier_feature_map(updates, n_components=27)
    
    # Project first, then embed
    print("Computing on PROJECTED updates...")
    projected_updates = random_projection(updates, 27)
    
    # Convert projected updates back to list format for embedding functions
    projected_as_list = []
    for i in range(projected_updates.shape[0]):
        # Create a single tensor per client (as if it's one parameter group)
        projected_as_list.append([torch.tensor(projected_updates[i])])
    
    proj_poly = polynomial_feature_map(projected_as_list, projection_dim=27)
    proj_rbf = rbf_feature_map(projected_as_list, n_components=27)
    proj_fourier = fourier_feature_map(projected_as_list, n_components=27)
    
    print(f"\nRaw update shapes:")
    print(f"  Polynomial: {raw_poly.shape}")
    print(f"  RBF: {raw_rbf.shape}")
    print(f"  Fourier: {raw_fourier.shape}")
    
    print(f"\nProjected update shapes:")
    print(f"  Polynomial: {proj_poly.shape}")
    print(f"  RBF: {proj_rbf.shape}")
    print(f"  Fourier: {proj_fourier.shape}")
    
    # Check if dimensions are reasonable
    max_acceptable_dim = 100  # Reasonable upper bound
    
    for name, shape in [("Raw Poly", raw_poly.shape), ("Raw RBF", raw_rbf.shape), 
                       ("Raw Fourier", raw_fourier.shape), ("Proj Poly", proj_poly.shape),
                       ("Proj RBF", proj_rbf.shape), ("Proj Fourier", proj_fourier.shape)]:
        if shape[1] > max_acceptable_dim:
            print(f"WARNING: {name} has {shape[1]} dimensions (> {max_acceptable_dim})")

def test_dimension_consistency():
    """Test that all methods produce consistent output dimensions"""
    print("\n=== Dimension Consistency Test ===")
    
    target_dim = 27
    updates = create_mock_updates(18, 10000)
    
    poly_out = polynomial_feature_map(updates, projection_dim=target_dim)
    rbf_out = rbf_feature_map(updates, n_components=target_dim)
    fourier_out = fourier_feature_map(updates, n_components=target_dim)
    
    print(f"Target dimension: {target_dim}")
    print(f"Polynomial output: {poly_out.shape[1]} (should be {target_dim})")
    print(f"RBF output: {rbf_out.shape[1]} (should be {target_dim})")
    print(f"Fourier output: {fourier_out.shape[1]} (should be {2*target_dim} for cos+sin)")
    
    # Check consistency
    assert poly_out.shape[1] == target_dim, f"Polynomial dim mismatch: {poly_out.shape[1]} != {target_dim}"
    assert rbf_out.shape[1] == target_dim, f"RBF dim mismatch: {rbf_out.shape[1]} != {target_dim}"
    assert fourier_out.shape[1] == 2*target_dim, f"Fourier dim mismatch: {fourier_out.shape[1]} != {2*target_dim}"
    
    print("✓ All dimension checks passed!")

if __name__ == "__main__":
    print("Testing Classical Non-Linear Embeddings")
    print("=" * 50)
    
    test_memory_usage()
    test_projected_vs_raw()
    test_dimension_consistency()
    
    print("\n" + "=" * 50)
    print("Testing complete!")