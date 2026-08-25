"""
Example: Training a Hybrid Quantum-Classical Neural Network
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.insert(0, '..')

from hqnn import HybridModel, train_hqnn
from utils import create_synthetic_data, plot_training_history


def main():
    """
    Main training example
    """
    # Configuration
    input_size = 10
    output_size = 2
    batch_size = 32
    epochs = 20
    learning_rate = 0.001
    
    quantum_config = {
        'num_qubits': 4,
        'num_parameters': 8
    }
    
    # Create synthetic data
    print("Creating synthetic data...")
    X_train, y_train = create_synthetic_data(num_samples=500, num_features=input_size)
    X_val, y_val = create_synthetic_data(num_samples=100, num_features=input_size)
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    print("Creating hybrid quantum-classical model...")
    model = HybridModel(
        input_size=input_size,
        quantum_config=quantum_config,
        output_size=output_size
    )
    
    # Train model
    print("Starting training...")
    history = train_hqnn(
        model,
        train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate
    )
    
    # Plot results
    print("Plotting results...")
    fig = plot_training_history(history, title='HQNN Training')
    fig.savefig('hqnn_training_history.png')
    print("Training completed! Results saved to 'hqnn_training_history.png'")


if __name__ == '__main__':
    main()