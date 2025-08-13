#!/usr/bin/env python3
# Simple end-to-end demonstration of Classical vs Quantum Projection for Byzantine detection
# in Federated Learning with CIFAR-10 dataset

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import random
import time
import os
import csv
import gc  # Add garbage collector import
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

# Import Qiskit with robust error handling
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap

# Robust simulator imports
# First try the standard qiskit-aer
try:
    from qiskit_aer import AerSimulator
    print("Using qiskit_aer.AerSimulator")
except ImportError:
    try:
        # Try older qiskit.aer location
        from qiskit.providers.aer import AerSimulator
        print("Using qiskit.providers.aer.AerSimulator")
    except ImportError:
        try:
            # Try direct import from qiskit
            from qiskit import Aer
            AerSimulator = Aer.get_backend
            print("Using qiskit.Aer for simulation")
        except ImportError:
            print("ERROR: Could not import any version of AerSimulator. Please check your Qiskit installation.")
            raise

# Robust sampler import (try different versions)
try:
    from qiskit.primitives import BackendSamplerV2 as Sampler  # Use V2 Sampler
    print("Using qiskit.primitives.BackendSamplerV2")
except ImportError:
    try:
        from qiskit.primitives import Sampler
        print("Using qiskit.primitives.Sampler")
    except ImportError:
        try:
            from qiskit_aer.primitives import Sampler
            print("Using qiskit_aer.primitives.Sampler")
        except ImportError:
            print("ERROR: Could not import Sampler. Please check your Qiskit installation.")
            raise

from qiskit.compiler import transpile

# === Hard-coded Parameters ===
NUM_HONEST = 15
NUM_BYZANTINE = 3
TOTAL_CLIENTS = NUM_HONEST + NUM_BYZANTINE
ROUNDS = 30
LOCAL_EPOCHS = 7
PROJECTION_DIM = 27 # For classical projection and Qiskit
PROJECTION_METHOD = "importance" # Changed to importance-weighted projection
BYZANTINE_ATTACK = "lie_attack"
DETECTION_NOISE = 0.5 # Noise added before distance calc for classical
DATASET = "mnist"
LEARNING_RATE = 0.07
BATCH_SIZE = 32
SEED = 42

# Classical Non-Linear Feature Map Parameters
POLY_DEGREE = 2
RBF_GAMMA = 0.5  # Reduced from 1.0 for better performance
FOURIER_SIGMA = 0.5  # Match RBF gamma for consistency

# === Setup ===
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Loading (CIFAR-10) ---
def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # CIFAR-10 normalization
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False)
    print(f"Loaded CIFAR-10: {len(trainset)} train, {len(testset)} test")
    return trainset, test_loader

# --- Simple CNN Model (CIFAR-10) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_weights(self):
        return [p.data.clone().cpu() for p in self.parameters()] # Keep weights on CPU

    def set_weights(self, weights):
        with torch.no_grad():
            for p, w in zip(self.parameters(), weights):
                p.copy_(w.to(p.device))

# --- Federated Learning Utilities ---
def partition_data(dataset, num_clients):
    samples_per_client = len(dataset) // num_clients
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    client_data_indices = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client
        client_data_indices.append(indices[start_idx:end_idx])
    return client_data_indices

def train_client(model, dataloader, epochs, lr):
    local_model = SimpleCNN().to(device) # Create a new instance
    local_model.load_state_dict(model.state_dict()) # Copy weights
    optimizer = optim.SGD(local_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    local_model.train()
    for _ in range(epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = local_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    # Return weight *difference* (update vector)
    with torch.no_grad():
        update = [(new_p.cpu() - old_p.cpu()) for old_p, new_p in zip(model.parameters(), local_model.parameters())]
    return update # Return update vectors on CPU

def generate_byzantine_update(honest_updates):
    """Lie attack: approximate opposite direction of the mean honest update"""
    if not honest_updates:
        return []
    num_params = len(honest_updates[0])
    mean_update = [torch.zeros_like(p, device='cpu') for p in honest_updates[0]]
    std_update = [torch.zeros_like(p, device='cpu') for p in honest_updates[0]]

    for i in range(num_params):
        # Move tensors to the active device for calculation
        param_stack = torch.stack([upd[i].to(device) for upd in honest_updates])
        mean_p = torch.mean(param_stack, dim=0)
        std_dev = torch.std(param_stack, dim=0)

        # Avoid division by zero or very small std dev
        std_p = torch.where(std_dev < 1e-6, torch.ones_like(std_dev) * 1e-6, std_dev)

        # Store results back on CPU
        mean_update[i] = mean_p.cpu()
        std_update[i] = std_p.cpu()

    # Use random z_max value like in multikrum_federated_mnist_cifar.py
    z_max_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    z_max = random.choice(z_max_options)

    print(f"  Using z_max={z_max} for Byzantine lie attack")

    # Calculation remains on CPU as inputs are now on CPU
    byzantine_update = []
    for i in range(num_params):
        malicious_param = mean_update[i] - z_max * std_update[i]
        byzantine_update.append(malicious_param)

    return byzantine_update

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def random_projection(updates, projection_dim):
    """ Project updates (list of weight lists) using a Gaussian random matrix. """
    flattened_updates = []
    for client_upd in updates:
        flat_upd = torch.cat([p.view(-1) for p in client_upd])
        flattened_updates.append(flat_upd)

    update_matrix = torch.stack(flattened_updates).to(device)
    original_dim = update_matrix.shape[1]

    projection_matrix = torch.randn(original_dim, projection_dim, device=device) / np.sqrt(projection_dim)
    projected_updates = torch.matmul(update_matrix, projection_matrix)

    return projected_updates.cpu().numpy() # Return projected updates on CPU

# --- Multi-KRUM Selection ---
def multi_krum(projected_grads, num_byzantine, detection_noise=0.0):
    """ Select clients using Multi-KRUM based on projected gradients. """
    n = projected_grads.shape[0]
    k = n - num_byzantine - 2 # Multi-KRUM parameter
    if k <= 0:
        print(f"Warning: k={k} is too small in Multi-KRUM (n={n}, f={num_byzantine}). Selecting all clients.")
        return list(range(n))

    # Add noise (optional, for classical comparison)
    if detection_noise > 0:
      noise = np.random.normal(0, detection_noise * np.std(projected_grads), projected_grads.shape)
      projected_grads = projected_grads + noise

    scores = np.zeros(n)
    distances = euclidean_distances(projected_grads, projected_grads) ** 2

    for i in range(n):
        # Find k closest neighbors (excluding self)
        neighbor_indices = np.argsort(distances[i, :])[1:k+1] # Exclude self (index 0)
        scores[i] = np.sum(distances[i, neighbor_indices])

    # Select n - f clients with the lowest scores
    num_to_select = n - num_byzantine
    selected_indices = np.argsort(scores)[:num_to_select]
    return sorted(list(selected_indices))

# --- Standard Multi-KRUM directly on updates ---
def standard_multikrum(updates, num_byzantine):
    """ Apply Multi-KRUM directly on flattened updates without projection. """
    # Flatten updates (list of weight lists) for distance calculation
    flattened_updates = []
    for client_upd in updates:
        flat_upd = torch.cat([p.view(-1) for p in client_upd]).to(device)
        flattened_updates.append(flat_upd)

    # Stack for batch processing and convert to numpy for distance calculation
    update_matrix = torch.stack(flattened_updates)
    update_numpy = update_matrix.cpu().numpy()

    n = len(updates)
    k = n - num_byzantine - 2 # Multi-KRUM parameter
    if k <= 0:
        print(f"Warning: k={k} is too small in Multi-KRUM (n={n}, f={num_byzantine}). Selecting all clients.")
        return list(range(n))

    # Compute distances directly on the original updates
    scores = np.zeros(n)
    distances = euclidean_distances(update_numpy, update_numpy) ** 2

    for i in range(n):
        # Find k closest neighbors (excluding self)
        neighbor_indices = np.argsort(distances[i, :])[1:k+1] # Exclude self
        scores[i] = np.sum(distances[i, neighbor_indices])

    # Select n - f clients with the lowest scores
    num_to_select = n - num_byzantine
    selected_indices = np.argsort(scores)[:num_to_select]
    return sorted(list(selected_indices))

# --- Quantum Embedding Utilities (Minimal) ---
def compute_quantum_embeddings(projected_grads, feature_map_type='zz', num_qubits=None):
    n_clients, proj_dim = projected_grads.shape
    num_qubits = num_qubits if num_qubits is not None else proj_dim
    if num_qubits > proj_dim:
        print(f"Warning: num_qubits ({num_qubits}) > projection_dim ({proj_dim}). Clipping to {proj_dim}.")
        num_qubits = proj_dim
    elif num_qubits < 2:
         print(f"Warning: num_qubits ({num_qubits}) too small. Setting to 2.")
         num_qubits = 2

    # *** IMPORTANT: MEMORY-EFFICIENT VERSION ***
    # Instead of storing the full 2^n dimensional vector (which is impossible for large n),
    # we'll store only the top-k most frequent measurement outcomes
    TOP_K = 1000  # Store only top 1000 measurement outcomes
    print(f"Computing {feature_map_type} quantum embeddings ({num_qubits} qubits) for {n_clients} clients...")
    print(f"Using memory-efficient representation with top {TOP_K} outcomes (instead of 2^{num_qubits} = {2**num_qubits})")

    # Create a matrix of shape (n_clients, TOP_K) instead of (n_clients, 2^num_qubits)
    embeddings = np.zeros((n_clients, TOP_K))
    basis_states = np.zeros((n_clients, TOP_K), dtype=np.int64)  # Store which states the values correspond to

    # Initialize simulator and sampler with robust GPU detection
    simulator = None
    sampler = None
    device_used = "Unknown" # Track which device is actually used
    gpu_available = False

    # First try to detect CUDA environment
    if torch.cuda.is_available():
        print(f"CUDA is available with {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

        # Check for cuQuantum
        try:
            import cuquantum
            print("cuQuantum is available for optimal GPU acceleration")
            cuquantum_available = True
        except ImportError:
            print("cuQuantum not found - GPU acceleration might be limited")
            cuquantum_available = False
    else:
        print("CUDA is not available, will use CPU for simulation")
        cuquantum_available = False

    # Attempt GPU initialization first if CUDA is available
    try:
        if torch.cuda.is_available():
            # Clean GPU memory before initialization
            torch.cuda.empty_cache()
            gc.collect()

            print("Attempting GPU-based AerSimulator initialization...")
            # For qiskit-aer-gpu, specify device='GPU'
            simulator = AerSimulator(device='GPU', method='statevector', max_memory_mb=20000)
            print("✓ Successfully initialized GPU-based AerSimulator with 20GB memory limit.")
            device_used = "GPU"
        else:
            print("CUDA not detected, initializing CPU-based AerSimulator...")
            simulator = AerSimulator(method="statevector")
            print("✓ Successfully initialized CPU-based AerSimulator.")
            device_used = "CPU"

        # Create the sampler based on the simulator
        try:
            # Try with BackendSamplerV2
            sampler = Sampler(backend=simulator)

            # Test with a simple circuit to verify it works
            test_circ = QuantumCircuit(2)
            test_circ.h(0)
            test_circ.cx(0, 1)
            test_circ.measure_all()
            test_job = sampler.run([test_circ])
            test_job.result()
            print(f"✓ Successfully initialized Sampler with {device_used}-based simulator backend.")
        except Exception as e1:
            print(f"BackendSamplerV2 failed: {e1}")
            # Try with regular Sampler
            try:
                sampler = Sampler(backend=simulator)
                # Different input format for older Sampler
                test_circ = QuantumCircuit(2)
                test_circ.h(0)
                test_circ.cx(0, 1)
                test_circ.measure_all()
                test_job = sampler.run([test_circ])
                test_job.result()
                print(f"✓ Successfully initialized Sampler with {device_used}-based simulator backend.")
            except Exception as e2:
                print(f"Regular Sampler also failed: {e2}")
                # Fall back to direct simulator usage
                print("Falling back to direct simulator usage")
                sampler = simulator
                device_used = "CPU" # Still CPU
                test_circ = QuantumCircuit(2)
                test_circ.h(0)
                test_circ.cx(0, 1)
                transpiled = transpile(test_circ, simulator)
                simulator.run(transpiled).result()
                print("✓ Successfully initialized direct simulator usage")

    except Exception as e:
        print(f"Error initializing simulator (GPU or CPU): {e}")
        print("Falling back to CPU simulation attempt if possible...")
        try:
            simulator = AerSimulator(method="statevector")
            # Re-attempt sampler creation with CPU simulator
            try:
                sampler = Sampler(backend=simulator)
                device_used = "CPU" # Definitely CPU now
                test_circ = QuantumCircuit(2)
                test_circ.h(0)
                test_circ.cx(0, 1)
                test_circ.measure_all()
                test_job = sampler.run([test_circ])
                test_job.result()
                print("✓ Successfully initialized Fallback CPU-based simulation with BackendSamplerV2")
            except Exception as e_sampler_cpu:
                print(f"Fallback Sampler creation failed: {e_sampler_cpu}")
                print("Attempting direct CPU simulator usage as final fallback...")
                sampler = simulator # Fallback to direct simulator
                device_used = "CPU" # Still CPU
                test_circ = QuantumCircuit(2)
                test_circ.h(0)
                test_circ.cx(0, 1)
                transpiled = transpile(test_circ, simulator)
                simulator.run(transpiled).result()
                print("✓ Successfully initialized direct CPU simulator usage as fallback")

        except Exception as e_cpu_fallback:
             print(f"Error initializing fallback CPU simulator: {e_cpu_fallback}")
             return None # Return None if we can't even initialize CPU simulation

    # If we made it here, we have a working simulator/sampler
    start_time = time.time()
    direct_simulator_mode = sampler == simulator  # Check if we're using direct simulator mode

    # Process clients in batches to reduce memory pressure
    batch_size = 3  # Adjust this based on your available memory

    for batch_start in range(0, n_clients, batch_size):
        batch_end = min(batch_start + batch_size, n_clients)
        print(f"Processing client batch {batch_start+1}-{batch_end} of {n_clients}")

        # Clean memory before processing batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        for i in range(batch_start, batch_end):
            client_grad = projected_grads[i, :num_qubits]  # Use first num_qubits dimensions

            # Normalize data for feature map
            norm = np.linalg.norm(client_grad)
            if norm > 0:
                client_grad_normalized = client_grad / norm
            else:
                client_grad_normalized = client_grad  # Avoid division by zero

            # Create feature map circuit
            if feature_map_type == 'zz':
                fm = ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement='full')
            elif feature_map_type == 'pauli':
                # More expressive PauliFeatureMap but with reduced complexity
                fm = PauliFeatureMap(
                    feature_dimension=num_qubits,
                    reps=2,  # Back to original repetitions
                    paulis=['X', 'Y', 'Z', 'XY', 'YZ'],  # Reduced set of Pauli terms
                    entanglement='full'  # Keeping full entanglement as requested
                )
            elif feature_map_type == 'heisenberg':
                # Heisenberg Hamiltonian feature map
                fm = PauliFeatureMap(
                    feature_dimension=num_qubits,
                    reps=2,
                    paulis=['XX', 'YY', 'ZZ'],  # Heisenberg interaction terms
                    entanglement='full'         # Full entanglement for expressive power
                )
            else:
                raise ValueError("Unsupported feature map type")

            qc = QuantumCircuit(num_qubits)
            qc.append(fm, range(num_qubits))
            qc.measure_all()

            try:
                # Different handling based on simulator mode
                if direct_simulator_mode:
                    # Transpile circuit for the simulator
                    transpiled_qc = transpile(qc, simulator)
                    # Create parameters list for the feature map
                    bind_parameters = {}
                    for j, param_value in enumerate(client_grad_normalized):
                        bind_parameters[fm.parameters[j]] = param_value
                    # Bind the parameters
                    bound_qc = transpiled_qc.bind_parameters(bind_parameters)
                    # Run directly with simulator
                    result = simulator.run(bound_qc, shots=1024).result()  # Back to 1024 shots as we need good statistics
                    counts = result.get_counts()
                else:
                    # Transpile circuit for the sampler backend
                    transpiled_qc = transpile(qc, sampler.backend)
                    # For BackendSamplerV2
                    try:
                        pub = (transpiled_qc, [client_grad_normalized])
                        job = sampler.run([pub], shots=1024)  # Back to 1024 shots for sufficient statistics
                        result = job.result()[0]
                        counts = result.data.meas.get_counts()
                    except (AttributeError, TypeError) as e:
                        # For regular Sampler
                        bound_qc = qc.bind_parameters(dict(zip(fm.parameters, client_grad_normalized)))
                        transpiled_bound_qc = transpile(bound_qc, sampler.backend)
                        job = sampler.run([transpiled_bound_qc], shots=1024)
                        result = job.result()
                        counts = result.quasi_dists[0]

                # Process counts to get only TOP_K most frequent outcomes
                sorted_counts = []
                for basis_state, count in counts.items():
                    if isinstance(basis_state, str):
                        idx = int(basis_state, 2)  # Convert binary string to int index
                    else:
                        idx = basis_state  # Already an integer index
                    sorted_counts.append((idx, count))

                # Sort by count (highest first) and take top-k
                sorted_counts.sort(key=lambda x: x[1], reverse=True)
                top_k_counts = sorted_counts[:min(TOP_K, len(sorted_counts))]

                # Store the top-k outcomes
                for k, (state_idx, count) in enumerate(top_k_counts):
                    basis_states[i, k] = state_idx
                    embeddings[i, k] = count

            except Exception as e:
                print(f"Error during quantum computation for client {i}: {e}")
                # Leave embedding as zeros for this client

            # Normalize embedding vector
            emb_norm = np.linalg.norm(embeddings[i])
            if emb_norm > 0:
                embeddings[i] /= emb_norm

            # Clean up memory after each client
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Force garbage collection
            del fm, qc, transpiled_qc
            gc.collect()

            # Print memory stats every few clients
            if torch.cuda.is_available() and i % 5 == 0:
                print(f"  Client {i} complete. GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    elapsed = time.time() - start_time
    print(f"Quantum embedding computation finished in {elapsed:.2f}s using {device_used}")

    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {'embeddings': embeddings, 'basis_states': basis_states}

def distance_multikrum_selector(embedding_data, num_byzantine):
    """ Select clients using Multi-KRUM based on embedding distances. """
    if embedding_data is None:
        print("Cannot perform Multi-KRUM selection: Embeddings not computed.")
        return []

    # Extract embeddings and basis states
    embeddings = embedding_data['embeddings']
    basis_states = embedding_data['basis_states']

    n = embeddings.shape[0]
    k = n - num_byzantine - 2
    if k <= 0:
        print(f"Warning: k={k} is too small in Multi-KRUM (n={n}, f={num_byzantine}). Selecting all clients.")
        return list(range(n))

    # Compute distances efficiently based on sparse representation
    scores = np.zeros(n)
    distances = np.zeros((n, n))

    # Compute pairwise distances
    for i in range(n):
        for j in range(i+1, n):
            # Compute distance between client i and j using sparse embeddings
            # Find common basis states
            common_mask_i = np.zeros(basis_states.shape[1], dtype=bool)
            common_mask_j = np.zeros(basis_states.shape[1], dtype=bool)

            for idx_i in range(basis_states.shape[1]):
                state_i = basis_states[i, idx_i]
                if state_i == 0 and embeddings[i, idx_i] == 0:  # Skip empty entries
                    continue

                for idx_j in range(basis_states.shape[1]):
                    state_j = basis_states[j, idx_j]
                    if state_i == state_j:
                        common_mask_i[idx_i] = True
                        common_mask_j[idx_j] = True
                        break

            # Calculate squared distance only on specified states
            dist_squared = 0.0

            # Add contribution of states in i not in j
            dist_squared += np.sum(embeddings[i, ~common_mask_i]**2)

            # Add contribution of states in j not in i
            dist_squared += np.sum(embeddings[j, ~common_mask_j]**2)

            # Add contribution of common states
            for idx_i in range(basis_states.shape[1]):
                if common_mask_i[idx_i]:
                    state_i = basis_states[i, idx_i]
                    # Find matching state in j
                    for idx_j in range(basis_states.shape[1]):
                        if basis_states[j, idx_j] == state_i:
                            diff = embeddings[i, idx_i] - embeddings[j, idx_j]
                            dist_squared += diff**2
                            break

            # Store in distance matrix (symmetric)
            distances[i, j] = dist_squared
            distances[j, i] = dist_squared

    for i in range(n):
        # Find k closest neighbors (excluding self)
        neighbor_indices = np.argsort(distances[i, :])[1:k+1] # Exclude self
        scores[i] = np.sum(distances[i, neighbor_indices])

    num_to_select = n - num_byzantine
    selected_indices = np.argsort(scores)[:num_to_select]
    return sorted(list(selected_indices))

# --- Logging & Visualization Utilities ---
import pandas as pd

# Global variable to store results in memory
results_df = pd.DataFrame(columns=[
    'Round', 'Aggregator', 'Attack', 'TP', 'FN', 'TN', 'FP', 'Byzantine_Count',
    'Byzantine_Rejection', 'Honest_Retention', 'F1_Score', 'Selected_Indices', 'Test_Accuracy'
])

def log_results(round_num, aggregator_name, metrics, selected_indices, accuracy, byzantine_count, attack_type=None):
    """Log results to the global dataframe without printing to stdout"""
    global results_df

    # Use BYZANTINE_ATTACK if attack_type not provided
    if attack_type is None:
        attack_type = BYZANTINE_ATTACK

    # Format selected indices as comma-delimited string in square brackets
    indices_str = '[' + ','.join(str(idx) for idx in selected_indices) + ']'

    # Create a new row
    new_row = {
        'Round': round_num + 1,
        'Aggregator': aggregator_name,
        'Attack': attack_type,
        'TP': metrics['TP'],
        'FN': metrics['FN'],
        'TN': metrics['TN'],
        'FP': metrics['FP'],
        'Byzantine_Count': byzantine_count,
        'Byzantine_Rejection': metrics['Recall'],
        'Honest_Retention': metrics['HonestRetention'],
        'F1_Score': metrics['F1'],
        'Selected_Indices': indices_str,
        'Test_Accuracy': accuracy
    }

    # Append to the DataFrame without printing
    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

def visualize_results(attack_type):
    """Generate visualizations using in-memory dataframe with inline rendering"""
    print("\n--- Creating Visualization Plots ---")

    # First, print the complete dataframe
    print("\n=== Complete Results Dataframe ===")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(results_df)

    # Filter for the specific attack type
    attack_df = results_df[results_df['Attack'] == attack_type]

    # Group by aggregator type
    aggregator_groups = attack_df.groupby('Aggregator')

    # Print summary for each aggregator
    print("\n=== Summary by Aggregator ===")
    for agg_name, agg_data in aggregator_groups:
        print(f"\n--- {agg_name} ---")
        print(f"Average F1 Score: {agg_data['F1_Score'].mean():.4f}")
        print(f"Average Byzantine Rejection: {agg_data['Byzantine_Rejection'].mean():.4f}")
        print(f"Average Honest Retention: {agg_data['Honest_Retention'].mean():.4f}")

    # Define colors and labels for different aggregators
    colors = {
        # Classical methods
        'classical_standard': '#1f77b4',         # Muted blue
        'classical_random': '#aec7e8',           # Light blue
        'classical_importance': '#ffbb78',       # Light orange
        
        # Classical non-linear feature maps
        'classical_polynomial': '#ff9999',       # Light red
        'classical_rbf': '#66b3ff',             # Light blue
        'classical_fourier': '#99ff99',         # Light green

        # ZZ methods
        'quantum_zz_random': '#2ca02c',          # Green
        'quantum_zz_importance': '#98df8a',      # Light green

        # Pauli methods
        'quantum_pauli_random': '#9467bd',       # Purple
        'quantum_pauli_importance': '#c5b0d5',   # Light purple

        # Heisenberg methods
        'quantum_heisenberg_random': '#d62728',     # Red
        'quantum_heisenberg_importance': '#ff7f0e', # Orange
    }

    labels = {
        # Classical methods
        'classical_standard': 'Classical Standard',
        'classical_random': f'Classical + Random Proj ({PROJECTION_DIM}D)',
        'classical_importance': f'Classical + Importance Proj ({PROJECTION_DIM}D)',
        
        # Classical non-linear feature maps
        'classical_polynomial': f'Classical + Polynomial Features ({PROJECTION_DIM}D)',
        'classical_rbf': f'Classical + RBF Features ({PROJECTION_DIM}D)',
        'classical_fourier': f'Classical + Fourier Features ({PROJECTION_DIM}D)',

        # ZZ methods
        'quantum_zz_random': 'Quantum ZZ + Random',
        'quantum_zz_importance': 'Quantum ZZ + Importance',

        # Pauli methods
        'quantum_pauli_random': 'Quantum Pauli + Random',
        'quantum_pauli_importance': 'Quantum Pauli + Importance',

        # Heisenberg methods
        'quantum_heisenberg_random': 'Quantum Heisenberg + Random',
        'quantum_heisenberg_importance': 'Quantum Heisenberg + Importance',
    }

    # Get unique rounds
    rounds = sorted(attack_df['Round'].unique())

    # Create plots for different metrics
    metrics_to_plot = {
        'Byzantine_Rejection': 'Byzantine Rejection Rate',
        'Honest_Retention': 'Honest Client Retention',
        'F1_Score': 'F1 Score',
        'Test_Accuracy': 'Test Accuracy'
    }

    for metric, title in metrics_to_plot.items():
        plt.figure(figsize=(14, 9))

        for agg_name, agg_df in aggregator_groups:
            if agg_name in colors:
                plt.plot(agg_df['Round'], agg_df[metric], 'o-',
                        color=colors[agg_name],
                        label=labels.get(agg_name, agg_name))

        plt.title(f'{title} ({DATASET.upper()}, {attack_type})', fontsize=16)
        plt.xlabel('Round', fontsize=12)

        if metric in ['Byzantine_Rejection', 'Honest_Retention', 'F1_Score']:
            plt.ylabel(title, fontsize=12)
            plt.ylim(-0.05, 1.05)
        else:
            plt.ylabel(f'{title} (%)', fontsize=12)

        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()

        # Save and show plot
        filename = f'plot_{metric.lower()}_{attack_type}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {filename}")
        plt.show()

    # Create a combined True Positives plot with incremental changes
    plt.figure(figsize=(14, 9))

    # Get TP for the baseline (Classical Standard)
    if 'classical_standard' in aggregator_groups.groups:
        baseline_df = aggregator_groups.get_group('classical_standard')
        baseline_tp = baseline_df['TP'].values
        plt.plot(baseline_df['Round'], baseline_tp, 'o-',
                color=colors['classical_standard'],
                label=labels['classical_standard'])
    else:
        baseline_tp = None
        print("Warning: Classical Standard (baseline) not found for TP plot")

    # Plot other aggregators with incremental annotations
    for agg_name, agg_df in aggregator_groups:
        if agg_name == 'classical_standard':
            continue

        if agg_name in colors:
            tp_values = agg_df['TP'].values
            plt.plot(agg_df['Round'], tp_values, 'o-',
                    color=colors[agg_name],
                    label=labels.get(agg_name, agg_name))

            # Add incremental annotations if baseline exists
            if baseline_tp is not None:
                for i, (round_num, row) in enumerate(agg_df.iterrows()):
                    if i < len(baseline_tp):  # Ensure we don't go beyond baseline length
                        increment = row['TP'] - baseline_tp[i]
                        if increment != 0:
                            plt.annotate(f"{'+' if increment > 0 else ''}{increment}",
                                        xy=(row['Round'], row['TP']),
                                        xytext=(5, 5), textcoords='offset points',
                                        fontsize=8, color=colors[agg_name])

    plt.title(f'Byzantine Detection: True Positives ({DATASET.upper()}, {attack_type})', fontsize=16)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('TP (Byzantine clients correctly rejected)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()

    # Save and show plot
    tp_filename = f'plot_true_positives_{attack_type}.png'
    plt.savefig(tp_filename, dpi=300, bbox_inches='tight')
    print(f"True Positives plot saved as: {tp_filename}")
    plt.show()

    print("\nVisualization complete.")

# --- Metrics and Aggregation ---
def calculate_metrics(selected_indices, num_honest, num_byzantine):
    n = num_honest + num_byzantine
    honest_indices = set(range(num_honest))
    byzantine_indices = set(range(num_honest, n))

    # Ensure selected indices are valid integers within range
    valid_selected_indices = []
    for idx in selected_indices:
        try:
            int_idx = int(idx) # Convert potential numpy ints
            if 0 <= int_idx < n:
                valid_selected_indices.append(int_idx)
            else:
                print(f"Warning: Selected index {idx} out of range [0, {n-1}] - skipping.")
        except (ValueError, TypeError):
            print(f"Warning: Invalid index {idx} found - skipping.")

    selected_set = set(valid_selected_indices)
    rejected_set = set(range(n)) - selected_set

    tp = len(rejected_set.intersection(byzantine_indices)) # Correctly rejected Byzantine
    fp = len(rejected_set.intersection(honest_indices))    # Incorrectly rejected Honest
    tn = len(selected_set.intersection(honest_indices))      # Correctly selected Honest
    fn = len(selected_set.intersection(byzantine_indices))   # Incorrectly selected Byzantine

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Byzantine rejection rate
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    honest_retention = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'Precision': precision, 'Recall': recall, 'F1': f1, 'HonestRetention': honest_retention}

def aggregate_updates(updates, selected_indices, global_weights):
    if not selected_indices:
        print("Warning: No clients selected, returning original weights.")
        return global_weights

    aggregated_update = [torch.zeros_like(p, device='cpu') for p in updates[0]]
    for idx in selected_indices:
        for i in range(len(aggregated_update)):
            aggregated_update[i] += updates[idx][i]

    avg_update = [upd / len(selected_indices) for upd in aggregated_update]

    # Apply aggregated update to global weights
    new_weights = [(global_weights[i] + avg_update[i]) for i in range(len(global_weights))]
    return new_weights

# === Main Execution ===
def main():
    print(f"--- Starting Federated Learning --- ")
    print(f" Config: {DATASET}, {NUM_HONEST}H+{NUM_BYZANTINE}B clients, {ROUNDS} rounds, {LOCAL_EPOCHS} epochs")
    print(f" Attack: {BYZANTINE_ATTACK}, Projection: {PROJECTION_METHOD}({PROJECTION_DIM}D)")
    print(f" Classical Noise: {DETECTION_NOISE}")

    trainset, test_loader = load_cifar10()
    global_model = SimpleCNN().to(device)
    global_weights = global_model.get_weights() # Keep global weights on CPU

    client_indices = partition_data(trainset, NUM_HONEST)
    client_loaders = [
        DataLoader(Subset(trainset, indices), batch_size=BATCH_SIZE, shuffle=True)
        for indices in client_indices
    ]

    for round_num in range(ROUNDS):
        print(f"\n--- Round {round_num + 1}/{ROUNDS} --- ")
        round_start_time = time.time()

        # 1. Client Training
        client_updates = []
        print(f" Training {NUM_HONEST} honest clients...")
        for i in range(NUM_HONEST):
            update = train_client(global_model, client_loaders[i], LOCAL_EPOCHS, LEARNING_RATE)
            client_updates.append(update)

        # 2. Generate Byzantine Updates
        print(f" Generating {NUM_BYZANTINE} Byzantine updates ({BYZANTINE_ATTACK})...")
        # Store honest updates temporarily to generate attack based only on them
        honest_only_updates = client_updates[:NUM_HONEST]
        for _ in range(NUM_BYZANTINE):
            byz_update = generate_byzantine_update(honest_only_updates)
            client_updates.append(byz_update)

        # Memory cleanup after updates creation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # --- Perform Detection & Selection for each method ---
        results = {}

        # 3a. Standard Classical Multi-KRUM (no projection)
        print(" Running Standard Multi-KRUM (no projection)...")
        standard_selected = standard_multikrum(client_updates, NUM_BYZANTINE)
        standard_metrics = calculate_metrics(standard_selected, NUM_HONEST, NUM_BYZANTINE)
        results['standard'] = {'selected': standard_selected, 'metrics': standard_metrics}
        print(f"  Standard Selected: {len(standard_selected)} clients. Recall: {standard_metrics['Recall']:.4f}, F1: {standard_metrics['F1']:.4f}")

        # Memory cleanup after standard Multi-KRUM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # 3b. Project updates based on method
        print(f" Projecting {TOTAL_CLIENTS} updates to {PROJECTION_DIM} dimensions...")

        # Random projection
        random_projected_updates = random_projection(client_updates, PROJECTION_DIM)

        # Memory cleanup after random projection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Importance weighted projection
        importance_projected_updates = importance_weighted_projection(client_updates, PROJECTION_DIM)

        # Memory cleanup after importance projection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # 4a. Classical Multi-KRUM with Random Projection
        print(f" Running Classical Multi-KRUM (Random {PROJECTION_DIM}D Projection)...")
        random_classical_selected = multi_krum(random_projected_updates, NUM_BYZANTINE, DETECTION_NOISE)
        random_classical_metrics = calculate_metrics(random_classical_selected, NUM_HONEST, NUM_BYZANTINE)
        print(f"  Classical Random Proj ({PROJECTION_DIM}D) Selected: {len(random_classical_selected)} clients. Recall: {random_classical_metrics['Recall']:.4f}, F1: {random_classical_metrics['F1']:.4f}")

        # 4b. Classical Multi-KRUM with Importance Weighted Projection
        print(f" Running Classical Multi-KRUM (Importance Weighted {PROJECTION_DIM}D Projection)...")
        importance_classical_selected = multi_krum(importance_projected_updates, NUM_BYZANTINE, DETECTION_NOISE)
        importance_classical_metrics = calculate_metrics(importance_classical_selected, NUM_HONEST, NUM_BYZANTINE)
        print(f"  Classical Importance Proj ({PROJECTION_DIM}D) Selected: {len(importance_classical_selected)} clients. Recall: {importance_classical_metrics['Recall']:.4f}, F1: {importance_classical_metrics['F1']:.4f}")

        # Memory cleanup after classical methods
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # 4c. Classical Multi-KRUM with Polynomial Features
        print(f" Running Classical Multi-KRUM (Polynomial Features - {PROJECTION_DIM}D)...")
        polynomial_features = polynomial_feature_map(client_updates, projection_dim=PROJECTION_DIM)
        polynomial_selected = multi_krum(polynomial_features, NUM_BYZANTINE, DETECTION_NOISE)
        polynomial_metrics = calculate_metrics(polynomial_selected, NUM_HONEST, NUM_BYZANTINE)
        print(f"  Classical Polynomial Features (degree {POLY_DEGREE}, {PROJECTION_DIM}D) Selected: {len(polynomial_selected)} clients. Recall: {polynomial_metrics['Recall']:.4f}, F1: {polynomial_metrics['F1']:.4f}")

        # Memory cleanup after polynomial features
        del polynomial_features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # 4d. Classical Multi-KRUM with RBF Features
        print(f" Running Classical Multi-KRUM (RBF Features - {PROJECTION_DIM}D)...")
        rbf_features = rbf_feature_map(client_updates, n_components=PROJECTION_DIM)
        rbf_selected = multi_krum(rbf_features, NUM_BYZANTINE, DETECTION_NOISE)
        rbf_metrics = calculate_metrics(rbf_selected, NUM_HONEST, NUM_BYZANTINE)
        print(f"  Classical RBF Features (gamma {RBF_GAMMA}, {PROJECTION_DIM}D) Selected: {len(rbf_selected)} clients. Recall: {rbf_metrics['Recall']:.4f}, F1: {rbf_metrics['F1']:.4f}")

        # Memory cleanup after RBF features
        del rbf_features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # 4e. Classical Multi-KRUM with Fourier Features
        print(f" Running Classical Multi-KRUM (Fourier Features - {PROJECTION_DIM * 2}D)...")
        fourier_features = fourier_feature_map(client_updates, n_components=PROJECTION_DIM)
        fourier_selected = multi_krum(fourier_features, NUM_BYZANTINE, DETECTION_NOISE)
        fourier_metrics = calculate_metrics(fourier_selected, NUM_HONEST, NUM_BYZANTINE)
        print(f"  Classical Fourier Features (sigma {FOURIER_SIGMA}, {fourier_features.shape[1]}D) Selected: {len(fourier_selected)} clients. Recall: {fourier_metrics['Recall']:.4f}, F1: {fourier_metrics['F1']:.4f}")

        # Memory cleanup after Fourier features
        del fourier_features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # 5a. Quantum ZZ Embedding with Random Projection
        print(f" Running Quantum Multi-KRUM (ZZ Embedding with Random Projection - {PROJECTION_DIM} Qubits)...")
        zz_random_embeddings = compute_quantum_embeddings(random_projected_updates, 'zz', PROJECTION_DIM)
        zz_random_selected = distance_multikrum_selector(zz_random_embeddings, NUM_BYZANTINE)
        zz_random_metrics = calculate_metrics(zz_random_selected, NUM_HONEST, NUM_BYZANTINE)
        print(f"  ZZ Random Selected: {len(zz_random_selected)} clients. Recall: {zz_random_metrics['Recall']:.4f}, F1: {zz_random_metrics['F1']:.4f}")

        # Delete embeddings and clean memory
        del zz_random_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # 5b. Quantum ZZ Embedding with Importance Projection
        print(f" Running Quantum Multi-KRUM (ZZ Embedding with Importance Projection - {PROJECTION_DIM} Qubits)...")
        zz_importance_embeddings = compute_quantum_embeddings(importance_projected_updates, 'zz', PROJECTION_DIM)
        zz_importance_selected = distance_multikrum_selector(zz_importance_embeddings, NUM_BYZANTINE)
        zz_importance_metrics = calculate_metrics(zz_importance_selected, NUM_HONEST, NUM_BYZANTINE)
        print(f"  ZZ Importance Selected: {len(zz_importance_selected)} clients. Recall: {zz_importance_metrics['Recall']:.4f}, F1: {zz_importance_metrics['F1']:.4f}")

        # Delete embeddings and clean memory
        del zz_importance_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # 6a. Quantum Pauli Embedding with Random Projection
        print(f" Running Quantum Multi-KRUM (Pauli Embedding with Random Projection - {PROJECTION_DIM} Qubits)...")
        pauli_random_embeddings = compute_quantum_embeddings(random_projected_updates, 'pauli', PROJECTION_DIM)
        pauli_random_selected = distance_multikrum_selector(pauli_random_embeddings, NUM_BYZANTINE)
        pauli_random_metrics = calculate_metrics(pauli_random_selected, NUM_HONEST, NUM_BYZANTINE)
        print(f"  Pauli Random Selected: {len(pauli_random_selected)} clients. Recall: {pauli_random_metrics['Recall']:.4f}, F1: {pauli_random_metrics['F1']:.4f}")

        # Delete embeddings and clean memory
        del pauli_random_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # 6b. Quantum Pauli Embedding with Importance Projection
        print(f" Running Quantum Multi-KRUM (Pauli Embedding with Importance Projection - {PROJECTION_DIM} Qubits)...")
        pauli_importance_embeddings = compute_quantum_embeddings(importance_projected_updates, 'pauli', PROJECTION_DIM)
        pauli_importance_selected = distance_multikrum_selector(pauli_importance_embeddings, NUM_BYZANTINE)
        pauli_importance_metrics = calculate_metrics(pauli_importance_selected, NUM_HONEST, NUM_BYZANTINE)
        print(f"  Pauli Importance Selected: {len(pauli_importance_selected)} clients. Recall: {pauli_importance_metrics['Recall']:.4f}, F1: {pauli_importance_metrics['F1']:.4f}")

        # Delete embeddings and clean memory
        del pauli_importance_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # 7a. Quantum Heisenberg Embedding with Random Projection
        print(f" Running Quantum Multi-KRUM (Heisenberg Embedding with Random Projection - {PROJECTION_DIM} Qubits)...")
        heisenberg_random_embeddings = compute_quantum_embeddings(random_projected_updates, 'heisenberg', PROJECTION_DIM)
        heisenberg_random_selected = distance_multikrum_selector(heisenberg_random_embeddings, NUM_BYZANTINE)
        heisenberg_random_metrics = calculate_metrics(heisenberg_random_selected, NUM_HONEST, NUM_BYZANTINE)
        print(f"  Heisenberg Random Selected: {len(heisenberg_random_selected)} clients. Recall: {heisenberg_random_metrics['Recall']:.4f}, F1: {heisenberg_random_metrics['F1']:.4f}")

        # Delete embeddings and clean memory
        del heisenberg_random_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # 7b. Quantum Heisenberg Embedding with Importance Projection
        print(f" Running Quantum Multi-KRUM (Heisenberg Embedding with Importance Projection - {PROJECTION_DIM} Qubits)...")
        heisenberg_importance_embeddings = compute_quantum_embeddings(importance_projected_updates, 'heisenberg', PROJECTION_DIM)
        heisenberg_importance_selected = distance_multikrum_selector(heisenberg_importance_embeddings, NUM_BYZANTINE)
        heisenberg_importance_metrics = calculate_metrics(heisenberg_importance_selected, NUM_HONEST, NUM_BYZANTINE)
        print(f"  Heisenberg Importance Selected: {len(heisenberg_importance_selected)} clients. Recall: {heisenberg_importance_metrics['Recall']:.4f}, F1: {heisenberg_importance_metrics['F1']:.4f}")

        # Delete embeddings and clean memory
        del heisenberg_importance_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Delete projected updates to free up memory
        del random_projected_updates
        del importance_projected_updates
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # 8. Aggregate using Standard Multi-KRUM results & Update Global Model
        print(" Aggregating using Standard Multi-KRUM selected clients...")
        global_weights = aggregate_updates(client_updates, standard_selected, global_weights)
        global_model.set_weights(global_weights)

        # 9. Test Model Accuracy
        current_accuracy = test_model(global_model, test_loader)
        print(f" Round {round_num + 1} Test Accuracy: {current_accuracy:.2f}%")

        # 10. Log Results for all methods
        log_results(round_num, 'classical_standard', standard_metrics, standard_selected, current_accuracy, NUM_BYZANTINE, BYZANTINE_ATTACK)

        # Classical with different projections
        log_results(round_num, 'classical_random', random_classical_metrics, random_classical_selected, current_accuracy, NUM_BYZANTINE, BYZANTINE_ATTACK)
        log_results(round_num, 'classical_importance', importance_classical_metrics, importance_classical_selected, current_accuracy, NUM_BYZANTINE, BYZANTINE_ATTACK)
        
        # Classical non-linear feature maps
        log_results(round_num, 'classical_polynomial', polynomial_metrics, polynomial_selected, current_accuracy, NUM_BYZANTINE, BYZANTINE_ATTACK)
        log_results(round_num, 'classical_rbf', rbf_metrics, rbf_selected, current_accuracy, NUM_BYZANTINE, BYZANTINE_ATTACK)
        log_results(round_num, 'classical_fourier', fourier_metrics, fourier_selected, current_accuracy, NUM_BYZANTINE, BYZANTINE_ATTACK)

        # ZZ with different projections
        log_results(round_num, 'quantum_zz_random', zz_random_metrics, zz_random_selected, current_accuracy, NUM_BYZANTINE, BYZANTINE_ATTACK)
        log_results(round_num, 'quantum_zz_importance', zz_importance_metrics, zz_importance_selected, current_accuracy, NUM_BYZANTINE, BYZANTINE_ATTACK)

        # Pauli with different projections
        log_results(round_num, 'quantum_pauli_random', pauli_random_metrics, pauli_random_selected, current_accuracy, NUM_BYZANTINE, BYZANTINE_ATTACK)
        log_results(round_num, 'quantum_pauli_importance', pauli_importance_metrics, pauli_importance_selected, current_accuracy, NUM_BYZANTINE, BYZANTINE_ATTACK)

        # Heisenberg with different projections
        log_results(round_num, 'quantum_heisenberg_random', heisenberg_random_metrics, heisenberg_random_selected, current_accuracy, NUM_BYZANTINE, BYZANTINE_ATTACK)
        log_results(round_num, 'quantum_heisenberg_importance', heisenberg_importance_metrics, heisenberg_importance_selected, current_accuracy, NUM_BYZANTINE, BYZANTINE_ATTACK)

        round_time = time.time() - round_start_time
        print(f" Round {round_num + 1} finished in {round_time:.2f}s")

        # Final memory cleanup after each round
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print("\n--- Federated Learning Finished ---")
    final_accuracy = test_model(global_model, test_loader)
    print(f"Final model accuracy after {ROUNDS} rounds: {final_accuracy:.2f}%")

    # Print the complete CSV table after all rounds
    print("\n=== Complete Results Table (CSV Format) ===")
    print("Round,Aggregator,Attack,TP,FN,TN,FP,Byzantine_Count,Byzantine_Rejection,Honest_Retention,F1_Score,Selected_Indices,Test_Accuracy")

    # Sort by Round and then by Aggregator for cleaner output
    sorted_df = results_df.sort_values(by=['Round', 'Aggregator'])

    # Print each row as CSV
    for _, row in sorted_df.iterrows():
        print(f"{row['Round']},{row['Aggregator']},{row['Attack']},{row['TP']},{row['FN']},{row['TN']},{row['FP']},{row['Byzantine_Count']},{row['Byzantine_Rejection']:.4f},{row['Honest_Retention']:.4f},{row['F1_Score']:.4f},\"{row['Selected_Indices']}\",{row['Test_Accuracy']:.2f}")

    # Generate visualizations
    visualize_results(BYZANTINE_ATTACK)

def importance_weighted_projection(updates, projection_dim):
    """Project updates using importance-weighted random projection.

    This method weighs parameters by their magnitude before projection,
    giving more importance to larger gradients which may be more indicative
    of the update direction.
    """
    # Flatten updates first
    flattened_updates = []
    for client_upd in updates:
        flat_upd = torch.cat([p.view(-1) for p in client_upd])
        flattened_updates.append(flat_upd)

    update_matrix = torch.stack(flattened_updates).to(device)
    original_dim = update_matrix.shape[1]

    # Calculate importance weights based on variance across clients
    variance = torch.var(update_matrix, dim=0)
    importance_weights = variance / (torch.sum(variance) + 1e-8)  # Normalize weights

    # Apply weights to projection matrix
    projection_matrix = torch.randn(original_dim, projection_dim, device=device)
    for i in range(original_dim):
        projection_matrix[i, :] *= torch.sqrt(importance_weights[i])

    # Normalize projection matrix columns
    for j in range(projection_dim):
        col_norm = torch.norm(projection_matrix[:, j])
        if col_norm > 0:
            projection_matrix[:, j] /= col_norm

    # Project the updates
    projected_updates = torch.matmul(update_matrix, projection_matrix)

    return projected_updates.cpu().numpy()  # Return projected updates on CPU

# --- Classical Non-Linear Feature Maps ---
def polynomial_feature_map(updates, degree=None, projection_dim=None):
    """Apply polynomial feature transformation to updates.
    
    Args:
        updates: List of client update vectors
        degree: Polynomial degree (uses POLY_DEGREE if None)
        projection_dim: Target dimension for final projection (optional)
    
    Returns:
        Transformed feature matrix
    """
    if degree is None:
        degree = POLY_DEGREE
        
    # Flatten updates first
    flattened_updates = []
    for client_upd in updates:
        flat_upd = torch.cat([p.view(-1) for p in client_upd])
        flattened_updates.append(flat_upd.cpu().numpy())
    
    update_matrix = np.stack(flattened_updates)
    
    # Improved normalization: standardize each feature dimension
    update_std = np.std(update_matrix, axis=0, keepdims=True)
    update_std = np.where(update_std < 1e-8, 1.0, update_std)  # Avoid division by zero
    update_matrix = update_matrix / update_std
    
    # Additional row-wise normalization to prevent overflow
    row_norms = np.linalg.norm(update_matrix, axis=1, keepdims=True)
    row_norms = np.where(row_norms < 1e-8, 1.0, row_norms)
    update_matrix = update_matrix / row_norms
    
    # Apply polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    poly_features = poly.fit_transform(update_matrix)
    
    print(f"Polynomial features (degree {degree}): {update_matrix.shape} -> {poly_features.shape}")
    
    # Always project if we have more features than target dimension
    if projection_dim is not None and poly_features.shape[1] > projection_dim:
        # Importance-weighted random projection for better preservation
        feature_importance = np.var(poly_features, axis=0)
        feature_weights = feature_importance / (np.sum(feature_importance) + 1e-8)
        
        np.random.seed(SEED)  # Reproducible projection
        projection_matrix = np.random.randn(poly_features.shape[1], projection_dim)
        
        # Weight the projection matrix
        for i in range(poly_features.shape[1]):
            projection_matrix[i, :] *= np.sqrt(feature_weights[i])
        
        # Normalize columns
        for j in range(projection_dim):
            col_norm = np.linalg.norm(projection_matrix[:, j])
            if col_norm > 0:
                projection_matrix[:, j] /= col_norm
        
        poly_features = np.dot(poly_features, projection_matrix)
        print(f"Importance-weighted projected to: {poly_features.shape}")
    
    return poly_features

def rbf_feature_map(updates, gamma=None, n_components=None):
    """Apply RBF (Radial Basis Function) kernel feature transformation.
    
    Args:
        updates: List of client update vectors
        gamma: RBF kernel parameter (uses RBF_GAMMA if None)
        n_components: Number of random features for approximation
    
    Returns:
        RBF-transformed feature matrix
    """
    if gamma is None:
        gamma = RBF_GAMMA
        
    # Flatten updates first
    flattened_updates = []
    for client_upd in updates:
        flat_upd = torch.cat([p.view(-1) for p in client_upd])
        flattened_updates.append(flat_upd.cpu().numpy())
    
    update_matrix = np.stack(flattened_updates)
    original_dim = update_matrix.shape[1]
    
    # Normalize input data for stable RBF computation
    update_mean = np.mean(update_matrix, axis=0, keepdims=True)
    update_std = np.std(update_matrix, axis=0, keepdims=True)
    update_std = np.where(update_std < 1e-8, 1.0, update_std)
    update_matrix = (update_matrix - update_mean) / update_std
    
    # Set default n_components if not provided
    if n_components is None:
        n_components = min(PROJECTION_DIM, original_dim)
    
    # Random Fourier Features approximation for RBF kernel
    # Sample random frequencies
    np.random.seed(SEED)  # For reproducibility
    random_weights = np.random.normal(0, np.sqrt(2 * gamma), (original_dim, n_components))
    random_bias = np.random.uniform(0, 2 * np.pi, n_components)
    
    # Compute random features: sqrt(2/n_components) * cos(X @ W + b)
    projection = np.dot(update_matrix, random_weights) + random_bias
    rbf_features = np.sqrt(2.0 / n_components) * np.cos(projection)
    
    print(f"RBF features (gamma={gamma}): {update_matrix.shape} -> {rbf_features.shape}")
    
    return rbf_features

def fourier_feature_map(updates, n_components=None, sigma=None):
    """Apply Random Fourier Features transformation.
    
    Args:
        updates: List of client update vectors
        n_components: Number of random Fourier components
        sigma: Gaussian kernel bandwidth parameter (uses FOURIER_SIGMA if None)
    
    Returns:
        Fourier-transformed feature matrix
    """
    if sigma is None:
        sigma = FOURIER_SIGMA
        
    # Flatten updates first
    flattened_updates = []
    for client_upd in updates:
        flat_upd = torch.cat([p.view(-1) for p in client_upd])
        flattened_updates.append(flat_upd.cpu().numpy())
    
    update_matrix = np.stack(flattened_updates)
    original_dim = update_matrix.shape[1]
    
    # Normalize input data for stable computation
    update_mean = np.mean(update_matrix, axis=0, keepdims=True)
    update_std = np.std(update_matrix, axis=0, keepdims=True)
    update_std = np.where(update_std < 1e-8, 1.0, update_std)
    update_matrix = (update_matrix - update_mean) / update_std
    
    # Set default n_components if not provided
    if n_components is None:
        n_components = PROJECTION_DIM
    
    # Random Fourier Features for Gaussian RBF kernel
    np.random.seed(SEED)  # For reproducibility
    
    # Sample random frequencies from Gaussian distribution
    random_weights = np.random.normal(0, 1.0/sigma, (original_dim, n_components))
    random_bias = np.random.uniform(0, 2 * np.pi, n_components)
    
    # Compute features: [cos(X @ W + b), sin(X @ W + b)]
    projection = np.dot(update_matrix, random_weights) + random_bias
    cos_features = np.cos(projection)
    sin_features = np.sin(projection)
    
    # Concatenate cos and sin features
    fourier_features = np.concatenate([cos_features, sin_features], axis=1)
    
    # Normalize appropriately
    fourier_features = fourier_features / np.sqrt(n_components)
    
    print(f"Fourier features (sigma={sigma}): {update_matrix.shape} -> {fourier_features.shape}")
    
    return fourier_features

if __name__ == "__main__":
    main()