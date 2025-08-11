"""Custom exceptions for the quantum Byzantine detection system."""

class ByzantineDetectionError(Exception):
    """Base exception for Byzantine detection system"""
    pass

class ConfigurationError(ByzantineDetectionError):
    """Configuration-related errors"""
    pass

class QuantumBackendError(ByzantineDetectionError):
    """Quantum backend initialization or execution errors"""
    pass

class ResourceExhaustedError(ByzantineDetectionError):
    """Resource exhaustion errors"""
    pass

class DetectionError(ByzantineDetectionError):
    """Detection algorithm errors"""
    pass