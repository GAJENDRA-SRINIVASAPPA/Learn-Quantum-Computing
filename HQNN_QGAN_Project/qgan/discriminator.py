"""
Classical Discriminator for QGAN
"""

import torch
import torch.nn as nn


class ClassicalDiscriminator(nn.Module):
    """
    Classical discriminator network for QGAN.
    Determines if data is real or generated.
    """
    
    def __init__(self, input_size, hidden_size=64):
        """
        Initialize discriminator
        
        Args:
            input_size (int): Size of input data
            hidden_size (int): Size of hidden layers
        """
        super(ClassicalDiscriminator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass through discriminator
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Probability of being real
        """
        return self.network(x)


class AdvancedDiscriminator(nn.Module):
    """
    Advanced discriminator with batch normalization and spectral normalization
    """
    
    def __init__(self, input_size, hidden_size=128):
        """
        Initialize advanced discriminator
        
        Args:
            input_size (int): Input dimension
            hidden_size (int): Hidden dimension
        """
        super(AdvancedDiscriminator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Discriminator output
        """
        return self.network(x)