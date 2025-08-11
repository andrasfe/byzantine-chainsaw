"""Byzantine detection algorithms."""

from .classical import *
from .quantum import *

__all__ = ['MultiKrumDetector', 'QuantumByzantineDetector']