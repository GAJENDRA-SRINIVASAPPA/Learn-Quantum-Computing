"""
Visualization utilities
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history, title='Training History'):
    """
    Plot training history
    
    Args:
        history (dict): Training history dictionary
        title (str): Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss')
    if 'gen_loss' in history:
        axes[0].plot(history['gen_loss'], label='Generator Loss')
    if 'disc_loss' in history:
        axes[0].plot(history['disc_loss'], label='Discriminator Loss')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{title} - Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Accuracy')
    if 'val_acc' in history:
        axes[1].plot(history['val_acc'], label='Val Accuracy')
    if 'real_acc' in history:
        axes[1].plot(history['real_acc'], label='Real Accuracy')
    if 'fake_acc' in history:
        axes[1].plot(history['fake_acc'], label='Fake Accuracy')
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{title} - Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    return fig


def plot_generated_samples(samples, title='Generated Samples'):
    """
    Plot generated samples (for image data)
    
    Args:
        samples (np.array): Generated samples
        title (str): Plot title
    """
    num_samples = min(len(samples), 16)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    
    for idx, ax in enumerate(axes.flat):
        if idx < num_samples:
            ax.imshow(samples[idx].reshape(28, 28), cmap='gray')
        ax.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig