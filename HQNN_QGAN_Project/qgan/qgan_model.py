"""
Quantum GAN Model combining Quantum Generator and Classical Discriminator
"""

import torch
import torch.nn as nn
from .generator import QuantumGenerator
from .discriminator import ClassicalDiscriminator, AdvancedDiscriminator


class QGAN(nn.Module):
    """
    Quantum Generative Adversarial Network
    """
    
    def __init__(self, num_qubits, num_gen_params, discriminator_input_size, 
                 discriminator_hidden=64, use_advanced=False):
        """
        Initialize QGAN
        
        Args:
            num_qubits (int): Number of qubits in generator
            num_gen_params (int): Number of generator parameters
            discriminator_input_size (int): Input size for discriminator
            discriminator_hidden (int): Hidden size for discriminator
            use_advanced (bool): Use advanced discriminator
        """
        super(QGAN, self).__init__()
        
        self.generator = QuantumGenerator(num_qubits, num_gen_params)
        
        if use_advanced:
            self.discriminator = AdvancedDiscriminator(
                discriminator_input_size, 
                discriminator_hidden
            )
        else:
            self.discriminator = ClassicalDiscriminator(
                discriminator_input_size,
                discriminator_hidden
            )
    
    def forward(self, real_data):
        """
        Forward pass
        
        Args:
            real_data (torch.Tensor): Real training data
            
        Returns:
            tuple: (generated_data, discriminator_output)
        """
        # Generate data
        generated = self.generator.forward()
        generated_tensor = torch.tensor(generated, dtype=torch.float32)
        
        # Discriminate
        disc_output = self.discriminator(real_data)
        
        return generated_tensor, disc_output
    
    def get_generator_params(self):
        """
        Get generator parameters
        
        Returns:
            np.array: Generator parameters
        """
        return self.generator.parameters
    
    def update_generator(self, gradients, learning_rate=0.001):
        """
        Update generator parameters
        
        Args:
            gradients (np.array): Gradients
            learning_rate (float): Learning rate
        """
        self.generator.update_parameters(gradients, learning_rate)