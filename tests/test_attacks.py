"""Tests for Byzantine attack implementations."""

import pytest
import torch
import numpy as np

from attacks.byzantine_attacks import LieAttack
from attacks.attack_factory import AttackFactory
from config.base_config import AttackConfig, AttackType
from core.interfaces import ClientUpdate

class TestLieAttack:
    """Test LieAttack implementation"""
    
    def test_attack_creation(self):
        """Test creating LieAttack"""
        config = AttackConfig(attack_type=AttackType.LIE)
        attack = LieAttack(config)
        assert attack.config == config
        
    def test_generate_byzantine_update(self):
        """Test generating Byzantine update"""
        config = AttackConfig(attack_type=AttackType.LIE)
        attack = LieAttack(config)
        
        # Create mock honest updates
        honest_updates = []
        for i in range(3):
            weights = [torch.randn(10), torch.randn(5)]
            update = ClientUpdate(
                client_id=i,
                weights=weights,
                metadata={'client_type': 'honest'},
                is_byzantine=False
            )
            honest_updates.append(update)
            
        # Create global weights
        global_weights = [torch.randn(10), torch.randn(5)]
        
        # Generate Byzantine update
        byz_update = attack.generate_byzantine_update(honest_updates, global_weights)
        
        assert byz_update.is_byzantine == True
        assert byz_update.client_id == -1
        assert len(byz_update.weights) == len(global_weights)
        assert 'z_max' in byz_update.metadata
        
    def test_empty_honest_updates(self):
        """Test handling empty honest updates list"""
        config = AttackConfig(attack_type=AttackType.LIE)
        attack = LieAttack(config)
        
        global_weights = [torch.randn(10), torch.randn(5)]
        
        byz_update = attack.generate_byzantine_update([], global_weights)
        
        assert byz_update.is_byzantine == True
        assert len(byz_update.weights) == len(global_weights)
        # Should be zero updates when no honest updates available
        for w in byz_update.weights:
            assert torch.allclose(w, torch.zeros_like(w))
            
    def test_attack_strength_scaling(self):
        """Test attack strength scaling"""
        config = AttackConfig(attack_type=AttackType.LIE, attack_strength=2.0)
        attack = LieAttack(config)
        
        # Create mock honest updates
        honest_updates = []
        for i in range(2):
            weights = [torch.ones(5)]  # Simple weights for predictable result
            update = ClientUpdate(
                client_id=i,
                weights=weights,
                metadata={'client_type': 'honest'},
                is_byzantine=False
            )
            honest_updates.append(update)
            
        global_weights = [torch.ones(5)]
        
        byz_update = attack.generate_byzantine_update(honest_updates, global_weights)
        
        # Attack strength should scale the result
        assert not torch.allclose(byz_update.weights[0], torch.ones(5))

class TestAttackFactory:
    """Test attack factory implementation"""
    
    def test_create_lie_attack(self):
        """Test creating LieAttack through factory"""
        config = AttackConfig(attack_type=AttackType.LIE)
        factory = AttackFactory()
        
        attack = factory.create(AttackType.LIE, config)
        assert isinstance(attack, LieAttack)
        
    def test_unsupported_attack_type(self):
        """Test error handling for unsupported attack types"""
        factory = AttackFactory()
        config = AttackConfig()
        
        # Create a non-existent attack type for testing
        with pytest.raises(ValueError, match="Unsupported attack type"):
            # This will fail because we only implemented LieAttack
            factory.create(AttackType.GAUSSIAN, config)
            
    def test_register_new_attack(self):
        """Test registering new attack type"""
        factory = AttackFactory()
        
        # Define dummy attack class
        class DummyAttack(LieAttack):
            pass
            
        # Register new attack
        factory.register_attack(AttackType.GAUSSIAN, DummyAttack)
        
        config = AttackConfig(attack_type=AttackType.GAUSSIAN)
        attack = factory.create(AttackType.GAUSSIAN, config)
        assert isinstance(attack, DummyAttack)