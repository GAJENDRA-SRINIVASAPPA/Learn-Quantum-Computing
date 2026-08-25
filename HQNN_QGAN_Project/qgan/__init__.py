"""
Quantum Generative Adversarial Network (QGAN) Module
"""

from .generator import QuantumGenerator
from .discriminator import ClassicalDiscriminator
from .qgan_model import QGAN
from .training import train_qgan

__all__ = ['QuantumGenerator', 'ClassicalDiscriminator', 'QGAN', 'train_qgan']