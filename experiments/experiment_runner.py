"""Main experiment orchestration with dependency injection."""

import logging
import time
import torch
from typing import Dict, List
from pathlib import Path

from config.experiment_config import ExperimentConfig
from data.dataset_loader import DatasetLoader
from data.federated_data import FederatedDataManager
from models.model_factory import ModelFactory
from attacks.attack_factory import AttackFactory
from detection.classical.multikrum import MultiKrumDetector
from detection.classical.projections import RandomProjection, ImportanceWeightedProjection
from detection.quantum.quantum_detector import QuantumByzantineDetector
from core.interfaces import ClientUpdate
from utils.memory_manager import MemoryManager
from metrics.evaluator import MetricsEvaluator

class ExperimentRunner:
    """Main experiment orchestration with dependency injection"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set random seeds
        self._set_seeds()
        
        # Initialize device
        self.device = self._setup_device()
        
        # Initialize components using factories
        self.model_factory = ModelFactory()
        self.attack_factory = AttackFactory()
        
        # Initialize utilities
        self.memory_manager = MemoryManager()
        self.metrics_evaluator = MetricsEvaluator()
        
        # Initialize data components
        self.dataset_loader = DatasetLoader()
        
    def _set_seeds(self):
        """Set random seeds for reproducibility"""
        import random
        import numpy as np
        
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
    
    def _setup_device(self) -> torch.device:
        """Setup compute device"""
        if self.config.device == "auto":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        
        self.logger.info(f"Using device: {device}")
        return device
    
    def run(self):
        """Run the complete experiment"""
        self.logger.info(f"Starting experiment: {self.config.experiment_name}")
        
        try:
            # Setup experiment
            self._setup_experiment()
            
            # Initialize results tracking
            results = []
            
            # Main federated learning rounds
            for round_num in range(self.config.federated.num_rounds):
                with self.memory_manager.managed_operation(f"Round {round_num + 1}"):
                    round_results = self._run_round(round_num)
                    results.extend(round_results)
                    
            # Generate final results
            self._generate_results(results)
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}", exc_info=True)
            raise
        finally:
            self._cleanup()
    
    def _setup_experiment(self):
        """Setup experiment components"""
        # Load dataset
        self.trainset, self.test_loader = self.dataset_loader.load_dataset(self.config.dataset)
        
        # Initialize global model
        self.global_model = self.model_factory.create(
            self.config.dataset, 
            device=str(self.device)
        )
        self.global_weights = self.global_model.get_weights()
        
        # Setup federated data
        self.fed_data_manager = FederatedDataManager(
            self.trainset, 
            self.config.federated.batch_size
        )
        
        # Partition data among honest clients
        client_indices = self.fed_data_manager.partition_data(
            self.config.federated.num_honest_clients
        )
        self.client_loaders = self.fed_data_manager.create_client_loaders(client_indices)
        
        # Initialize attack strategy
        self.attack_strategy = self.attack_factory.create(
            self.config.attack.attack_type,
            self.config.attack
        )
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Experiment setup complete")
    
    def _run_round(self, round_num: int) -> List[Dict]:
        """Execute a single federated learning round"""
        self.logger.info(f"Round {round_num + 1}/{self.config.federated.num_rounds}")
        round_start_time = time.time()
        
        # 1. Train honest clients
        honest_updates = self._train_honest_clients()
        
        # 2. Generate Byzantine updates
        byzantine_updates = self._generate_byzantine_updates(honest_updates)
        
        # 3. Combine all updates
        all_updates = honest_updates + byzantine_updates
        
        # 4. Run detection methods
        detection_results = self._run_detection_methods(all_updates)
        
        # 5. Aggregate updates (using best method or standard)
        selected_indices = detection_results.get('standard_multikrum', {}).get('selected_indices', [])
        self._aggregate_and_update(all_updates, selected_indices)
        
        # 6. Evaluate model
        accuracy = self.global_model.evaluate(self.test_loader, self.device)
        
        # 7. Log results
        round_results = self._log_round_results(round_num, detection_results, accuracy)
        
        round_time = time.time() - round_start_time
        self.logger.info(f"Round {round_num + 1} finished in {round_time:.2f}s")
        
        return round_results
    
    def _train_honest_clients(self) -> List[ClientUpdate]:
        """Train all honest clients"""
        honest_updates = []
        
        self.logger.info(f"Training {self.config.federated.num_honest_clients} honest clients...")
        
        for i in range(self.config.federated.num_honest_clients):
            update = self._train_client(i, self.client_loaders[i])
            client_update = ClientUpdate(
                client_id=i,
                weights=update,
                metadata={'client_type': 'honest'},
                is_byzantine=False
            )
            honest_updates.append(client_update)
        
        return honest_updates
    
    def _train_client(self, client_id: int, dataloader) -> List[torch.Tensor]:
        """Train a single client"""
        # Create local model copy
        local_model = self.model_factory.create(self.config.dataset, device=str(self.device))
        local_model.set_weights(self.global_weights)
        
        # Set up optimizer
        optimizer = torch.optim.SGD(
            local_model.parameters(), 
            lr=self.config.federated.learning_rate
        )
        criterion = torch.nn.CrossEntropyLoss()
        
        # Local training
        local_model.train()
        for _ in range(self.config.federated.local_epochs):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = local_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # Return weight difference (update vector)
        with torch.no_grad():
            update = [
                (new_p.cpu() - old_p.cpu()) 
                for old_p, new_p in zip(self.global_weights, local_model.get_weights())
            ]
        
        return update
    
    def _generate_byzantine_updates(self, honest_updates: List[ClientUpdate]) -> List[ClientUpdate]:
        """Generate Byzantine updates"""
        byzantine_updates = []
        
        self.logger.info(f"Generating {self.config.federated.num_byzantine_clients} Byzantine updates...")
        
        for i in range(self.config.federated.num_byzantine_clients):
            byzantine_update = self.attack_strategy.generate_byzantine_update(
                honest_updates, 
                self.global_weights
            )
            byzantine_update.client_id = self.config.federated.num_honest_clients + i
            byzantine_updates.append(byzantine_update)
        
        return byzantine_updates
    
    def _run_detection_methods(self, all_updates: List[ClientUpdate]) -> Dict:
        """Run all configured detection methods"""
        results = {}
        
        # Standard Multi-KRUM
        standard_detector = MultiKrumDetector()
        selected = standard_detector.detect(all_updates, self.config.federated.num_byzantine_clients)
        metrics = self.metrics_evaluator.calculate_metrics(
            selected, 
            self.config.federated.num_honest_clients,
            self.config.federated.num_byzantine_clients
        )
        results['standard_multikrum'] = {'selected_indices': selected, 'metrics': metrics}
        
        # Classical with projections
        if 'classical_random' in [method.value for method in self.config.detection_methods]:
            random_proj = RandomProjection(device=str(self.device))
            random_detector = MultiKrumDetector(
                projection_strategy=random_proj,
                projection_dim=self.config.detection.projection_dim
            )
            selected = random_detector.detect(all_updates, self.config.federated.num_byzantine_clients)
            metrics = self.metrics_evaluator.calculate_metrics(
                selected,
                self.config.federated.num_honest_clients,
                self.config.federated.num_byzantine_clients
            )
            results['classical_random'] = {'selected_indices': selected, 'metrics': metrics}
        
        if 'classical_importance' in [method.value for method in self.config.detection_methods]:
            importance_proj = ImportanceWeightedProjection(device=str(self.device))
            importance_detector = MultiKrumDetector(
                projection_strategy=importance_proj,
                projection_dim=self.config.detection.projection_dim
            )
            selected = importance_detector.detect(all_updates, self.config.federated.num_byzantine_clients)
            metrics = self.metrics_evaluator.calculate_metrics(
                selected,
                self.config.federated.num_honest_clients,
                self.config.federated.num_byzantine_clients
            )
            results['classical_importance'] = {'selected_indices': selected, 'metrics': metrics}
        
        # Quantum methods
        quantum_methods = ['quantum_zz', 'quantum_pauli', 'quantum_heisenberg']
        for method_name in quantum_methods:
            if method_name in [method.value for method in self.config.detection_methods]:
                feature_map_type = method_name.replace('quantum_', '')
                
                # Try with random projection
                random_proj = RandomProjection(device=str(self.device))
                quantum_detector = QuantumByzantineDetector(
                    feature_map_type=feature_map_type,
                    quantum_config=self.config.quantum,
                    projection_strategy=random_proj,
                    projection_dim=self.config.detection.projection_dim
                )
                
                try:
                    selected = quantum_detector.detect(all_updates, self.config.federated.num_byzantine_clients)
                    metrics = self.metrics_evaluator.calculate_metrics(
                        selected,
                        self.config.federated.num_honest_clients,
                        self.config.federated.num_byzantine_clients
                    )
                    results[f'{method_name}_random'] = {'selected_indices': selected, 'metrics': metrics}
                except Exception as e:
                    self.logger.error(f"Quantum method {method_name} failed: {e}")
                    # Use classical fallback
                    selected = standard_detector.detect(all_updates, self.config.federated.num_byzantine_clients)
                    metrics = self.metrics_evaluator.calculate_metrics(
                        selected,
                        self.config.federated.num_honest_clients,
                        self.config.federated.num_byzantine_clients
                    )
                    results[f'{method_name}_random'] = {'selected_indices': selected, 'metrics': metrics}
        
        return results
    
    def _aggregate_and_update(self, updates: List[ClientUpdate], selected_indices: List[int]):
        """Aggregate selected updates and update global model"""
        if not selected_indices:
            self.logger.warning("No clients selected, keeping original weights")
            return
        
        # Average selected updates
        aggregated_update = [torch.zeros_like(p, device='cpu') for p in updates[0].weights]
        for idx in selected_indices:
            for i in range(len(aggregated_update)):
                aggregated_update[i] += updates[idx].weights[i]
        
        avg_update = [upd / len(selected_indices) for upd in aggregated_update]
        
        # Apply to global weights
        self.global_weights = [
            (self.global_weights[i] + avg_update[i]) 
            for i in range(len(self.global_weights))
        ]
        
        # Update global model
        self.global_model.set_weights(self.global_weights)
    
    def _log_round_results(self, round_num: int, detection_results: Dict, accuracy: float) -> List[Dict]:
        """Log results for the current round"""
        round_results = []
        
        for method_name, result in detection_results.items():
            round_result = {
                'round': round_num + 1,
                'method': method_name,
                'selected_indices': result['selected_indices'],
                'metrics': result['metrics'],
                'accuracy': accuracy,
                'attack_type': self.config.attack.attack_type.value
            }
            round_results.append(round_result)
            
            # Log key metrics
            self.logger.info(
                f"{method_name}: Selected {len(result['selected_indices'])} clients, "
                f"F1={result['metrics']['F1']:.4f}, "
                f"Byzantine Rejection={result['metrics']['Recall']:.4f}"
            )
        
        self.logger.info(f"Round {round_num + 1} Test Accuracy: {accuracy:.2f}%")
        
        return round_results
    
    def _generate_results(self, results: List[Dict]):
        """Generate and save final results"""
        self.logger.info("Generating final results...")
        
        # TODO: Implement results visualization and saving
        # This would include CSV export, plots, etc.
        
        # Log final accuracy
        final_accuracy = self.global_model.evaluate(self.test_loader, self.device)
        self.logger.info(f"Final model accuracy: {final_accuracy:.2f}%")
    
    def _cleanup(self):
        """Cleanup resources"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()