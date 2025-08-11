"""Quantum feature map implementations."""

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from typing import Dict, Any
import numpy as np

from config.base_config import QuantumConfig

class QuantumFeatureMapBuilder:
    """Builder for quantum feature maps"""
    
    def __init__(self, config: QuantumConfig):
        self.config = config
        
    def build_feature_map(
        self,
        feature_map_type: str,
        num_qubits: int,
        reps: int = None
    ) -> Any:
        """Build a quantum feature map of the specified type"""
        if reps is None:
            reps = self.config.feature_map_reps
            
        if feature_map_type.lower() == 'zz':
            return ZZFeatureMap(
                feature_dimension=num_qubits,
                reps=reps,
                entanglement=self.config.entanglement
            )
            
        elif feature_map_type.lower() == 'pauli':
            return PauliFeatureMap(
                feature_dimension=num_qubits,
                reps=reps,
                paulis=['X', 'Y', 'Z', 'XY', 'YZ'],
                entanglement=self.config.entanglement
            )
            
        elif feature_map_type.lower() == 'heisenberg':
            return PauliFeatureMap(
                feature_dimension=num_qubits,
                reps=reps,
                paulis=['XX', 'YY', 'ZZ'],
                entanglement=self.config.entanglement
            )
            
        else:
            raise ValueError(f"Unsupported feature map type: {feature_map_type}")
    
    def create_circuit(
        self,
        feature_map_type: str,
        parameters: np.ndarray,
        num_qubits: int
    ) -> QuantumCircuit:
        """Create a quantum circuit with the feature map"""
        # Build the feature map
        feature_map = self.build_feature_map(feature_map_type, num_qubits)
        
        # Create quantum circuit
        qc = QuantumCircuit(num_qubits)
        qc.append(feature_map, range(num_qubits))
        qc.measure_all()
        
        return qc, feature_map