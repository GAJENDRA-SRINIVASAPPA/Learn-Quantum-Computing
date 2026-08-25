"""
Data loading utilities
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np


def load_data(dataset_name='mnist', train=True):
    """
    Load standard datasets
    
    Args:
        dataset_name (str): Name of dataset ('mnist', 'cifar10')
        train (bool): Load training or test data
        
    Returns:
        dataset: PyTorch dataset
    """
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616)
            )
        ])
        dataset = datasets.CIFAR10(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def create_dataloaders(train_dataset, test_dataset, batch_size=32, num_workers=4):
    """
    Create PyTorch dataloaders
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        batch_size (int): Batch size
        num_workers (int): Number of workers
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader


def create_synthetic_data(num_samples=1000, num_features=10):
    """
    Create synthetic data for testing
    
    Args:
        num_samples (int): Number of samples
        num_features (int): Number of features
        
    Returns:
        tuple: (X, y) as tensors
    """
    X = torch.randn(num_samples, num_features)
    y = torch.randint(0, 2, (num_samples,))
    
    return X, y