"""Memory management utilities."""

import gc
import logging
from contextlib import contextmanager
from typing import Dict

class MemoryManager:
    """Manages memory for both classical and quantum operations"""
    
    def __init__(self, threshold_gb: float = 10.0):
        self.threshold_gb = threshold_gb
        self.logger = logging.getLogger(__name__)
        
    def check_memory(self) -> Dict[str, float]:
        """Check current memory usage"""
        memory_info = {}
        
        try:
            import psutil
            memory_info.update({
                'cpu_percent': psutil.virtual_memory().percent,
                'cpu_available_gb': psutil.virtual_memory().available / (1024**3)
            })
        except ImportError:
            self.logger.warning("psutil not available for CPU memory monitoring")
        
        try:
            import torch
            if torch.cuda.is_available():
                memory_info.update({
                    'gpu_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                    'gpu_reserved_gb': torch.cuda.memory_reserved() / (1024**3)
                })
        except Exception:
            pass
            
        return memory_info
    
    @contextmanager
    def managed_operation(self, operation_name: str):
        """Context manager for memory-intensive operations"""
        self.logger.debug(f"Starting {operation_name}")
        initial_memory = self.check_memory()
        
        try:
            yield
        finally:
            self._cleanup()
            final_memory = self.check_memory()
            self._log_memory_delta(operation_name, initial_memory, final_memory)
    
    def _cleanup(self):
        """Perform memory cleanup"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
    
    def _log_memory_delta(self, operation: str, initial: Dict, final: Dict):
        """Log memory usage changes"""
        if 'cpu_available_gb' in initial and 'cpu_available_gb' in final:
            cpu_delta = initial['cpu_available_gb'] - final['cpu_available_gb']
            self.logger.debug(f"{operation} CPU memory delta: {cpu_delta:.2f} GB")
        
        if 'gpu_allocated_gb' in initial and 'gpu_allocated_gb' in final:
            gpu_delta = final['gpu_allocated_gb'] - initial['gpu_allocated_gb']
            self.logger.debug(f"{operation} GPU memory delta: {gpu_delta:.2f} GB")