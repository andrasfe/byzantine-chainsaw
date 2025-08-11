"""Quantum embedding computation with memory management."""

import time
import gc
import logging
from typing import Dict, Optional
import numpy as np

# Robust Qiskit imports with error handling
try:
    from qiskit_aer import AerSimulator
    print("Using qiskit_aer.AerSimulator")
except ImportError:
    try:
        from qiskit.providers.aer import AerSimulator
        print("Using qiskit.providers.aer.AerSimulator")
    except ImportError:
        from qiskit import Aer
        AerSimulator = Aer.get_backend
        print("Using qiskit.Aer for simulation")

# Sampler imports
try:
    from qiskit.primitives import BackendSamplerV2 as Sampler
    print("Using qiskit.primitives.BackendSamplerV2")
except ImportError:
    try:
        from qiskit.primitives import Sampler
        print("Using qiskit.primitives.Sampler")
    except ImportError:
        from qiskit_aer.primitives import Sampler
        print("Using qiskit_aer.primitives.Sampler")

from qiskit.compiler import transpile

from config.base_config import QuantumConfig
from .feature_maps import QuantumFeatureMapBuilder

class QuantumEmbeddingComputer:
    """Computes quantum embeddings with memory optimization"""
    
    def __init__(self, config: QuantumConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.simulator = None
        self.sampler = None
        
    def compute_embedding(
        self,
        projected_data: np.ndarray,
        feature_map_type: str,
        num_qubits: Optional[int] = None
    ) -> Optional[Dict[str, np.ndarray]]:
        """Compute quantum embeddings for projected gradient data"""
        
        n_clients, proj_dim = projected_data.shape
        num_qubits = num_qubits if num_qubits is not None else proj_dim
        
        # Validate and adjust num_qubits
        if num_qubits > proj_dim:
            self.logger.warning(f"num_qubits ({num_qubits}) > projection_dim ({proj_dim}). Clipping to {proj_dim}.")
            num_qubits = proj_dim
        elif num_qubits < 2:
            self.logger.warning(f"num_qubits ({num_qubits}) too small. Setting to 2.")
            num_qubits = 2

        # Initialize memory-efficient storage
        TOP_K = self.config.top_k_outcomes
        embeddings = np.zeros((n_clients, TOP_K))
        basis_states = np.zeros((n_clients, TOP_K), dtype=np.int64)
        
        self.logger.info(f"Computing {feature_map_type} quantum embeddings ({num_qubits} qubits) for {n_clients} clients...")
        self.logger.info(f"Using memory-efficient representation with top {TOP_K} outcomes")
        
        # Initialize quantum backend
        if not self._initialize_quantum_backend():
            self.logger.error("Failed to initialize quantum backend")
            return None
        
        # Create feature map builder
        feature_map_builder = QuantumFeatureMapBuilder(self.config)
        
        start_time = time.time()
        direct_simulator_mode = self.sampler == self.simulator
        
        # Process clients in batches
        for batch_start in range(0, n_clients, self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, n_clients)
            self.logger.info(f"Processing client batch {batch_start+1}-{batch_end} of {n_clients}")
            
            self._cleanup_memory()
            
            for i in range(batch_start, batch_end):
                client_grad = projected_data[i, :num_qubits]
                
                # Normalize data for feature map
                norm = np.linalg.norm(client_grad)
                if norm > 0:
                    client_grad_normalized = client_grad / norm
                else:
                    client_grad_normalized = client_grad
                
                # Create quantum circuit
                try:
                    qc, feature_map = feature_map_builder.create_circuit(
                        feature_map_type, client_grad_normalized, num_qubits
                    )
                    
                    # Execute circuit and get counts
                    counts = self._execute_circuit(
                        qc, feature_map, client_grad_normalized, direct_simulator_mode
                    )
                    
                    if counts:
                        # Process counts to get top-K outcomes
                        self._process_counts(counts, i, embeddings, basis_states, TOP_K)
                    
                except Exception as e:
                    self.logger.error(f"Error processing client {i}: {e}")
                    # Leave embedding as zeros for this client
                
                # Normalize embedding vector
                emb_norm = np.linalg.norm(embeddings[i])
                if emb_norm > 0:
                    embeddings[i] /= emb_norm
                
                # Cleanup after each client
                self._cleanup_memory()
                
                if i % 5 == 0:
                    self._log_memory_stats(i)
        
        elapsed = time.time() - start_time
        self.logger.info(f"Quantum embedding computation finished in {elapsed:.2f}s")
        
        # Final cleanup
        self._cleanup_memory()
        
        return {'embeddings': embeddings, 'basis_states': basis_states}
    
    def _initialize_quantum_backend(self) -> bool:
        """Initialize quantum simulator and sampler"""
        try:
            # Try GPU first if available
            import torch
            if self.config.use_gpu and torch.cuda.is_available():
                self.logger.info("Attempting GPU-based AerSimulator initialization...")
                try:
                    self.simulator = AerSimulator(
                        device='GPU', 
                        method='statevector', 
                        max_memory_mb=self.config.max_memory_mb
                    )
                    self.logger.info("✓ Successfully initialized GPU-based AerSimulator")
                except Exception as e:
                    self.logger.warning(f"GPU initialization failed: {e}, falling back to CPU")
                    self.simulator = AerSimulator(method="statevector")
            else:
                self.logger.info("Initializing CPU-based AerSimulator...")
                self.simulator = AerSimulator(method="statevector")
                
            # Initialize sampler
            try:
                self.sampler = Sampler(backend=self.simulator)
                # Test with simple circuit
                from qiskit import QuantumCircuit
                test_circ = QuantumCircuit(2)
                test_circ.h(0)
                test_circ.cx(0, 1)
                test_circ.measure_all()
                test_job = self.sampler.run([test_circ])
                test_job.result()
                self.logger.info("✓ Successfully initialized Sampler")
                
            except Exception as e:
                self.logger.warning(f"Sampler initialization failed: {e}, using direct simulator")
                self.sampler = self.simulator
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize quantum backend: {e}")
            return False
    
    def _execute_circuit(self, qc, feature_map, parameters, direct_mode) -> Optional[Dict]:
        """Execute quantum circuit and return measurement counts"""
        try:
            if direct_mode:
                # Direct simulator mode
                transpiled_qc = transpile(qc, self.simulator)
                bind_parameters = dict(zip(feature_map.parameters, parameters))
                bound_qc = transpiled_qc.bind_parameters(bind_parameters)
                result = self.simulator.run(bound_qc, shots=self.config.num_shots).result()
                return result.get_counts()
            else:
                # Sampler mode
                transpiled_qc = transpile(qc, self.sampler.backend)
                try:
                    # Try BackendSamplerV2 format
                    pub = (transpiled_qc, [parameters])
                    job = self.sampler.run([pub], shots=self.config.num_shots)
                    result = job.result()[0]
                    return result.data.meas.get_counts()
                except (AttributeError, TypeError):
                    # Regular Sampler format
                    bound_qc = qc.bind_parameters(dict(zip(feature_map.parameters, parameters)))
                    transpiled_bound_qc = transpile(bound_qc, self.sampler.backend)
                    job = self.sampler.run([transpiled_bound_qc], shots=self.config.num_shots)
                    result = job.result()
                    return result.quasi_dists[0]
                    
        except Exception as e:
            self.logger.error(f"Circuit execution failed: {e}")
            return None
    
    def _process_counts(self, counts, client_idx, embeddings, basis_states, top_k):
        """Process measurement counts and store top-k outcomes"""
        sorted_counts = []
        for basis_state, count in counts.items():
            if isinstance(basis_state, str):
                idx = int(basis_state, 2)  # Convert binary string to int
            else:
                idx = basis_state  # Already an integer
            sorted_counts.append((idx, count))

        # Sort by count (highest first) and take top-k
        sorted_counts.sort(key=lambda x: x[1], reverse=True)
        top_k_counts = sorted_counts[:min(top_k, len(sorted_counts))]

        # Store the top-k outcomes
        for k, (state_idx, count) in enumerate(top_k_counts):
            basis_states[client_idx, k] = state_idx
            embeddings[client_idx, k] = count
    
    def _cleanup_memory(self):
        """Perform memory cleanup"""
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _log_memory_stats(self, client_idx):
        """Log memory statistics"""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                self.logger.info(f"Client {client_idx} complete. GPU memory allocated: {allocated:.2f} GB")
        except Exception:
            pass