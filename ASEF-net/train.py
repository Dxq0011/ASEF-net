"""
Training script for SEF-Net v4
SEF-Net: Learning Spatial Exposure Flows for Low-Light Image Enhancement
"""

import os
import sys
import time
import argparse
import yaml
from datetime import datetime
import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import numpy as np

# Performance optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from models.sef_net_v4 import SEFNet, SEFNetLoss
from data.dataset import LOLv1Dataset, LOLv2Dataset
from utils.metrics import calculate_psnr, calculate_ssim


def get_dataset(config):
    """Create dataset based on config."""
    dataset_name = config['dataset']['name']
    root = config['dataset']['root']
    img_size = config['dataset'].get('img_size', 256)
    
    if dataset_name == 'lol_v1':
        train_dataset = LOLv1Dataset(root, split='train', img_size=img_size, augment=True)
        val_dataset = LOLv1Dataset(root, split='test', img_size=img_size, augment=False)
    elif dataset_name == 'lol_v2':
        subset = config['dataset'].get('subset', 'Real')
        train_dataset = LOLv2Dataset(root, subset=subset, split='train', img_size=img_size, augment=True)
        val_dataset = LOLv2Dataset(root, subset=subset, split='test', img_size=img_size, augment=False)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_dataset, val_dataset


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, global_step):
    """Train for one epoch."""
    model.train()
    
    epoch_losses = {
        'total': 0.0,
        'rec': 0.0,
        'mono': 0.0,
        'range': 0.0,
        'sep': 0.0,
        'perc': 0.0,
        'color': 0.0,
        'sat': 0.0,
        'contrast': 0.0,
        'tone': 0.0,
        'edge': 0.0
    }
    
    for batch_idx, batch in enumerate(dataloader):
        I_low = batch['low'].to(device)
        I_gt = batch['gt'].to(device)
        
        # Forward pass
        I_out, v, n, c, trajectory = model(I_low)
        
        # Compute loss
        loss, loss_dict = criterion(I_out, I_gt, trajectory, v, I_low)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        for key in epoch_losses:
            if key in loss_dict:
                epoch_losses[key] += loss_dict[key]
        
        # Logging
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss_dict['total']:.4f} "
                  f"(Rec: {loss_dict['loss_rec']:.4f}, "
                  f"Perc: {loss_dict.get('loss_perc', 0):.4f}, "
                  f"Color: {loss_dict.get('loss_color', 0):.4f}, "
                  f"Sat: {loss_dict.get('loss_sat', 0):.4f}, "
                  f"Tone: {loss_dict.get('loss_tone', 0):.4f}, "
                  f"Edge: {loss_dict.get('loss_edge', 0):.4f})")
        
        # TensorBoard logging
        if writer is not None and batch_idx % 50 == 0:
            writer.add_scalar('Train/Loss_total', loss_dict['total'], global_step)
            writer.add_scalar('Train/Loss_rec', loss_dict['loss_rec'], global_step)
            writer.add_scalar('Train/Loss_mono', loss_dict['loss_mono'], global_step)
            writer.add_scalar('Train/Loss_sep', loss_dict['loss_sep'], global_step)
            writer.add_scalar('Train/Loss_perc', loss_dict.get('loss_perc', 0), global_step)
            writer.add_scalar('Train/Loss_color', loss_dict.get('loss_color', 0), global_step)
            writer.add_scalar('Train/Loss_sat', loss_dict.get('loss_sat', 0), global_step)
            writer.add_scalar('Train/Loss_contrast', loss_dict.get('loss_contrast', 0), global_step)
            writer.add_scalar('Train/Loss_tone', loss_dict.get('loss_tone', 0), global_step)
            writer.add_scalar('Train/Loss_edge', loss_dict.get('loss_edge', 0), global_step)
        
        global_step += 1
    
    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= len(dataloader)
    
    return epoch_losses, global_step


def validate(model, dataloader, device, epoch, writer):
    """Validate on validation set."""
    model.eval()
    
    psnr_list = []
    ssim_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            I_low = batch['low'].to(device)
            I_gt = batch['gt'].to(device)
            
            # Forward pass
            I_out, v, n, c, trajectory = model(I_low)
            
            # Clamp to [0, 1]
            I_out = torch.clamp(I_out, 0, 1)
            
            # Calculate metrics
            for i in range(I_out.shape[0]):
                psnr = calculate_psnr(I_out[i], I_gt[i])
                ssim = calculate_ssim(I_out[i], I_gt[i])
                psnr_list.append(psnr)
                ssim_list.append(ssim)
            
            # Save visualization for first batch
            if batch_idx == 0 and writer is not None:
                # Create comparison grid
                comparison = torch.cat([I_low[:4], I_out[:4], I_gt[:4]], dim=0)
                grid = make_grid(comparison, nrow=4, normalize=False)
                writer.add_image('Validation/Comparison', grid, epoch)
                
                # Visualize fields
                v_vis = v[:1].repeat(1, 3, 1, 1)  # Repeat to 3 channels
                v_vis = torch.nn.functional.interpolate(v_vis, size=(256, 256), mode='bilinear')
                writer.add_image('Validation/Velocity', v_vis[0], epoch)
    
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    
    print(f"Validation - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
    
    if writer is not None:
        writer.add_scalar('Val/PSNR', avg_psnr, epoch)
        writer.add_scalar('Val/SSIM', avg_ssim, epoch)
    
    return avg_psnr, avg_ssim


def main():
    parser = argparse.ArgumentParser(description='Train SEF-Net v4')
    parser.add_argument('--config', type=str, default='configs/sef_net_v4.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from. If not set, auto-resume from checkpoints/<exp_name>/latest.pth if exists')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset, val_dataset = get_dataset(config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    model = SEFNet(
        in_ch=config['model']['in_ch'],
        T=config['model']['T'],
        C_emb=config['model']['C_emb'],
        feat_ch=config['model']['feat_ch'],
        ode_steps=config['model']['ode_steps'],
        epsilon=config['model']['epsilon']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss
    criterion = SEFNetLoss(
        lambda_rec=config['loss']['lambda_rec'],
        lambda_perc=config['loss'].get('lambda_perc', 0.1),
        lambda_phys=config['loss']['lambda_phys'],
        lambda_prior=config['loss']['lambda_prior'],
        lambda_temp=config['loss'].get('lambda_temp', 0.01),
        lambda_color=config['loss'].get('lambda_color', 0.05),
        lambda_sat=config['loss'].get('lambda_sat', 0.05),
        lambda_contrast=config['loss'].get('lambda_contrast', 0.05),
        lambda_tone=config['loss'].get('lambda_tone', 0.08),
        lambda_edge=config['loss'].get('lambda_edge', 0.05),
        device=device
    )
    
    # Create optimizer
    lr = float(config['training']['lr'])
    weight_decay = float(config['training'].get('weight_decay', 1e-4))
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=1e-6
    )
    
    # Setup checkpoint directory first (needed for auto-resume)
    exp_name = config.get('experiment_name', 'sef_net_v4')
    checkpoint_dir = os.path.join('checkpoints', exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Resume from checkpoint
    start_epoch = 0
    best_psnr = 0.0
    global_step = 0
    
    # Auto-resume: if --resume not set, check for latest.pth
    resume_path = args.resume
    if resume_path is None:
        auto_resume = os.path.join(checkpoint_dir, 'latest.pth')
        if os.path.exists(auto_resume):
            resume_path = auto_resume
            print(f"Found checkpoint: {resume_path}, auto-resuming...")
    
    if resume_path is not None and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        # Use strict=False to allow loading partial weights when architecture changes
        missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing:
            print(f"Warning: Missing keys in checkpoint (new layers initialized randomly): {missing}")
        if unexpected:
            print(f"Warning: Unexpected keys in checkpoint (ignored): {unexpected}")
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}. Starting with fresh optimizer.")
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint.get('best_psnr', 0.0)
        global_step = checkpoint.get('global_step', 0)
        print(f"Resumed from epoch {start_epoch}, best PSNR: {best_psnr:.4f}")
    
    # Setup tensorboard
    log_dir = os.path.join('logs', exp_name, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    # Use absolute path to avoid TF gfile issues
    log_dir = os.path.abspath(log_dir)
    print(f"TensorBoard log dir: {log_dir}")
    try:
        writer = SummaryWriter(log_dir)
    except Exception as e:
        print(f"Warning: Failed to create SummaryWriter: {e}")
        print("Continuing without TensorBoard logging.")
        writer = None
    
    # Training loop
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}")
        
        # Train
        epoch_losses, global_step = train_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, writer, global_step
        )
        
        print(f"Epoch {epoch} - Avg Loss: {epoch_losses['total']:.4f}")
        
        # Validate
        if (epoch + 1) % config['training'].get('val_every', 5) == 0:
            avg_psnr, avg_ssim = validate(model, val_loader, device, epoch, writer)
            
            # Save best model
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_psnr': best_psnr,
                    'config': config
                }, os.path.join(checkpoint_dir, 'best.pth'))
                print(f"Saved best model with PSNR: {best_psnr:.4f}")
        
        # Save latest checkpoint (every epoch for resume capability)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_psnr': best_psnr,
            'global_step': global_step,
            'config': config
        }, os.path.join(checkpoint_dir, 'latest.pth'))
        
        # Also save periodic checkpoint
        if (epoch + 1) % config['training'].get('save_every', 10) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_psnr': best_psnr,
                'global_step': global_step,
                'config': config
            }, os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth'))
            print(f"Saved checkpoint at epoch {epoch+1}")
        
        # Step scheduler
        scheduler.step()
    
    if writer is not None:
        writer.close()
    print(f"\nTraining completed! Best PSNR: {best_psnr:.4f}")


if __name__ == '__main__':
    main()
