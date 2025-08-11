"""Factory for creating attack strategies."""

from typing import Dict, Type

from core.interfaces import AttackStrategy
from config.base_config import AttackType, AttackConfig
from .byzantine_attacks import LieAttack

class AttackFactory:
    """Factory for creating attack strategies"""
    
    def __init__(self):
        self._attacks: Dict[AttackType, Type[AttackStrategy]] = {
            AttackType.LIE: LieAttack,
        }
    
    def create(self, attack_type: AttackType, config: AttackConfig) -> AttackStrategy:
        """Create an attack strategy"""
        if attack_type not in self._attacks:
            raise ValueError(f"Unsupported attack type: {attack_type}")
        
        attack_class = self._attacks[attack_type]
        return attack_class(config)
    
    def register_attack(self, attack_type: AttackType, attack_class: Type[AttackStrategy]):
        """Register a new attack class"""
        self._attacks[attack_type] = attack_class