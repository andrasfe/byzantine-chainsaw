"""Byzantine attack implementations."""

import torch
import random
import numpy as np
from typing import List

from core.interfaces import AttackStrategy, ClientUpdate
from config.base_config import AttackConfig

class LieAttack(AttackStrategy):
    """Lie attack: generates updates opposite to the mean honest update"""
    
    def __init__(self, config: AttackConfig):
        self.config = config
        self.device = 'cpu'  # Keep computations on CPU for consistency
    
    def generate_byzantine_update(
        self, 
        honest_updates: List[ClientUpdate],
        global_weights: List[torch.Tensor]
    ) -> ClientUpdate:
        """Generate Byzantine update using lie attack strategy"""
        if not honest_updates:
            # Return zero update if no honest updates available
            return ClientUpdate(
                client_id=-1,
                weights=[torch.zeros_like(w) for w in global_weights],
                metadata={'attack_type': 'lie_attack', 'z_max': 0.0},
                is_byzantine=True
            )
        
        # Extract weight updates from honest clients
        honest_weight_updates = [update.weights for update in honest_updates]
        num_params = len(honest_weight_updates[0])
        
        # Calculate mean and standard deviation of honest updates
        mean_update = [torch.zeros_like(p, device=self.device) for p in honest_weight_updates[0]]
        std_update = [torch.zeros_like(p, device=self.device) for p in honest_weight_updates[0]]
        
        for i in range(num_params):
            # Stack parameters from all honest clients
            param_stack = torch.stack([upd[i].to(self.device) for upd in honest_weight_updates])
            mean_p = torch.mean(param_stack, dim=0)
            std_dev = torch.std(param_stack, dim=0)
            
            # Avoid division by zero
            std_p = torch.where(std_dev < 1e-6, torch.ones_like(std_dev) * 1e-6, std_dev)
            
            mean_update[i] = mean_p.cpu()
            std_update[i] = std_p.cpu()
        
        # Select random z_max value
        z_max = random.choice(self.config.z_max_range)
        
        # Generate Byzantine update: mean - z_max * std
        byzantine_weights = []
        for i in range(num_params):
            malicious_param = mean_update[i] - z_max * std_update[i]
            byzantine_weights.append(malicious_param * self.config.attack_strength)
        
        return ClientUpdate(
            client_id=-1,  # Special ID for Byzantine clients
            weights=byzantine_weights,
            metadata={'attack_type': 'lie_attack', 'z_max': z_max},
            is_byzantine=True
        )