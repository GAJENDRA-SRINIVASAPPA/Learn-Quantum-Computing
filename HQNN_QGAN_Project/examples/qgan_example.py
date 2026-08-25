"""
Example: Training a Quantum Generative Adversarial Network
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.insert(0, '..')

from qgan import QGAN, train_qgan
from utils import create_synthetic_data, plot_training_history


def main():
    """
    Main QGAN training example
    """
    # Configuration
    num_qubits = 4
    num_gen_params = 8
    discriminator_input_size = 4
    batch_size = 32
    epochs = 50
    gen_lr = 0.0002
    disc_lr = 0.0002
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create synthetic data
    print("Creating synthetic data...")
    X, y = create_synthetic_data(
        num_samples=500,
        num_features=discriminator_input_size
    )
    
    # Create dataloader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create QGAN model
    print("Creating QGAN model...")
    qgan = QGAN(
        num_qubits=num_qubits,
        num_gen_params=num_gen_params,
        discriminator_input_size=discriminator_input_size,
        discriminator_hidden=64,
        use_advanced=True
    )
    
    # Train QGAN
    print("Starting QGAN training...")
    history = train_qgan(
        qgan,
        dataloader,
        epochs=epochs,
        gen_lr=gen_lr,
        disc_lr=disc_lr,
        device=device
    )
    
    # Plot results
    print("Plotting results...")
    fig = plot_training_history(history, title='QGAN Training')
    fig.savefig('qgan_training_history.png')
    print("Training completed! Results saved to 'qgan_training_history.png'")


if __name__ == '__main__':
    main()