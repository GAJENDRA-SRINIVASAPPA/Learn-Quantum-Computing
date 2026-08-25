"""
Hybrid Quantum-Classical Neural Network Model
"""

import torch
import torch.nn as nn
from .quantum_layer import QuantumLayer


class HybridModel(nn.Module):
    """
    A hybrid neural network that integrates quantum and classical layers.
    """
    
    def __init__(self, input_size, quantum_config, output_size=2):
        """
        Initialize the Hybrid Model
        
        Args:
            input_size (int): Size of input data
            quantum_config (dict): Configuration for quantum layer
                - num_qubits: Number of qubits
                - num_parameters: Number of parameters
            output_size (int): Size of output
        """
        super(HybridModel, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Classical preprocessing layer
        self.classical_input = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, quantum_config['num_qubits'])
        )
        
        # Quantum layer
        self.quantum_layer = QuantumLayer(
            num_qubits=quantum_config['num_qubits'],
            num_parameters=quantum_config['num_parameters']
        )
        
        # Classical postprocessing layer
        self.classical_output = nn.Sequential(
            nn.Linear(quantum_config['num_qubits'], 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        """
        Forward pass through the hybrid model
        
        Args:
            x (torch.Tensor): Input data
            
        Returns:
            torch.Tensor: Model output
        """
        # Classical preprocessing
        x = self.classical_input(x)
        
        # Convert to numpy for quantum layer
        x_numpy = x.detach().cpu().numpy()
        
        # Quantum processing
        quantum_output = []
        for sample in x_numpy:
            output = self.quantum_layer.forward(sample)
            quantum_output.append(output)
        
        # Convert back to tensor
        x = torch.tensor(quantum_output, dtype=torch.float32)
        
        # Classical postprocessing
        x = self.classical_output(x)
        
        return x
    
    def get_quantum_parameters(self):
        """
        Get quantum layer parameters
        
        Returns:
            np.array: Quantum parameters
        """
        return self.quantum_layer.parameters


class HybridVariationalModel(nn.Module):
    """
    Advanced hybrid model with multiple quantum layers and skip connections
    """
    
    def __init__(self, input_size, num_quantum_layers=2, quantum_config=None, output_size=2):
        """
        Initialize advanced hybrid model
        
        Args:
            input_size (int): Input dimension
            num_quantum_layers (int): Number of quantum layers
            quantum_config (dict): Quantum configuration
            output_size (int): Output dimension
        """
        super(HybridVariationalModel, self).__init__()
        
        if quantum_config is None:
            quantum_config = {
                'num_qubits': 4,
                'num_parameters': 8
            }
        
        self.input_projection = nn.Linear(input_size, quantum_config['num_qubits'])
        
        self.quantum_layers = nn.ModuleList([
            QuantumLayer(
                num_qubits=quantum_config['num_qubits'],
                num_parameters=quantum_config['num_parameters']
            )
            for _ in range(num_quantum_layers)
        ])
        
        self.output_layer = nn.Linear(quantum_config['num_qubits'], output_size)
    
    def forward(self, x):
        """
        Forward pass with multiple quantum layers
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output predictions
        """
        x = self.input_projection(x)
        
        for quantum_layer in self.quantum_layers:
            x_quantum = quantum_layer.forward(x.detach().cpu().numpy())
            x = torch.tensor(x_quantum, dtype=torch.float32)
        
        x = self.output_layer(x)
        return x