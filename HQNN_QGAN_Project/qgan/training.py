"""
Training loop for Quantum GAN
"""

import torch
import torch.nn as nn
import torch.optim as optim


def train_qgan(qgan_model, real_data_loader, epochs=100, 
               gen_lr=0.001, disc_lr=0.001, device='cpu'):
    """
    Train QGAN model
    
    Args:
        qgan_model: QGAN model instance
        real_data_loader: DataLoader for real data
        epochs (int): Number of training epochs
        gen_lr (float): Generator learning rate
        disc_lr (float): Discriminator learning rate
        device: Device to train on
        
    Returns:
        dict: Training history
    """
    
    qgan_model.to(device)
    
    # Optimizers
    gen_optimizer = optim.Adam(
        qgan_model.generator.parameters if hasattr(qgan_model.generator, 'parameters') 
        else [], 
        lr=gen_lr, 
        betas=(0.5, 0.999)
    )
    
    disc_optimizer = optim.Adam(
        qgan_model.discriminator.parameters(),
        lr=disc_lr,
        betas=(0.5, 0.999)
    )
    
    # Loss function
    criterion = nn.BCELoss()
    
    history = {
        'gen_loss': [],
        'disc_loss': [],
        'real_acc': [],
        'fake_acc': []
    }
    
    for epoch in range(epochs):
        gen_loss_epoch = 0
        disc_loss_epoch = 0
        real_acc_epoch = 0
        fake_acc_epoch = 0
        
        for batch_idx, (real_data, _) in enumerate(real_data_loader):
            real_data = real_data.to(device)
            batch_size = real_data.size(0)
            
            # Labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # ====== Train Discriminator ======
            disc_optimizer.zero_grad()
            
            # Real data
            real_output = qgan_model.discriminator(real_data)
            real_loss = criterion(real_output, real_labels)
            
            # Fake data
            fake_data = torch.tensor(
                [qgan_model.generator.forward() for _ in range(batch_size)],
                dtype=torch.float32
            ).to(device)
            fake_output = qgan_model.discriminator(fake_data.detach())
            fake_loss = criterion(fake_output, fake_labels)
            
            disc_loss = real_loss + fake_loss
            disc_loss.backward()
            disc_optimizer.step()
            
            # ====== Train Generator ======
            gen_optimizer.zero_grad()
            
            fake_data = torch.tensor(
                [qgan_model.generator.forward() for _ in range(batch_size)],
                dtype=torch.float32
            ).to(device)
            fake_output = qgan_model.discriminator(fake_data)
            gen_loss = criterion(fake_output, real_labels)  # Trick: use real labels
            
            gen_loss.backward()
            gen_optimizer.step()
            
            # Metrics
            gen_loss_epoch += gen_loss.item()
            disc_loss_epoch += disc_loss.item()
            real_acc_epoch += (real_output > 0.5).float().mean().item()
            fake_acc_epoch += (fake_output < 0.5).float().mean().item()
        
        # Average metrics
        num_batches = len(real_data_loader)
        history['gen_loss'].append(gen_loss_epoch / num_batches)
        history['disc_loss'].append(disc_loss_epoch / num_batches)
        history['real_acc'].append(real_acc_epoch / num_batches)
        history['fake_acc'].append(fake_acc_epoch / num_batches)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Gen Loss: {history['gen_loss'][-1]:.4f} | "
                  f"Disc Loss: {history['disc_loss'][-1]:.4f} | "
                  f"Real Acc: {history['real_acc'][-1]:.4f} | "
                  f"Fake Acc: {history['fake_acc'][-1]:.4f}")
    
    return history