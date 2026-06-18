"""
Testing/Evaluation script for SEF-Net v4
SEF-Net: Learning Spatial Exposure Flows for Low-Light Image Enhancement
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import yaml

from models.sef_net_v4 import SEFNet
from data.dataset import LOLv1Dataset, LOLv2Dataset
from utils.metrics import calculate_psnr, calculate_ssim


def get_dataset(config, split='test', img_size=None):
    """Create dataset based on config.
    img_size: None = keep original resolution, int = resize to that size.
    """
    dataset_name = config['dataset']['name']
    root = config['dataset']['root']
    # img_size is explicitly passed; if None, dataset keeps original resolution
    
    if dataset_name == 'lol_v1':
        test_dataset = LOLv1Dataset(root, split=split, img_size=img_size, augment=False)
    elif dataset_name == 'lol_v2':
        subset = config['dataset'].get('subset', 'Real')
        test_dataset = LOLv2Dataset(root, subset=subset, split=split, img_size=img_size, augment=False)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return test_dataset


def enhance_image_tiled(model, image_tensor, device, tile_size=2048, overlap=256):
    """
    Process a large image in overlapping tiles to avoid OOM.
    Uses reflect padding and linear feathering in overlap regions.
    
    Args:
        model: SEFNet model
        image_tensor: [B, 3, H, W], B must be 1
        device: torch.device
        tile_size: max tile dimension (will be rounded to multiple of 8)
        overlap: overlap between tiles (will be rounded to multiple of 8)
    
    Returns:
        I_out: [B, 3, H, W] enhanced image
    """
    assert image_tensor.shape[0] == 1, "Tile-based inference only supports batch_size=1"
    
    _, _, H, W = image_tensor.shape
    
    # If image is small enough, process directly
    if H <= tile_size and W <= tile_size:
        with torch.no_grad():
            I_out, _, _, _, _ = model(image_tensor)
        return torch.clamp(I_out, 0, 1)
    
    # Ensure tile_size and overlap are multiples of 8 (model has 3 stride-2 layers)
    tile_size = (tile_size // 8) * 8
    overlap = (overlap // 8) * 8
    
    # Pad image so that tiles fit evenly
    stride = tile_size - overlap
    pad_h = (stride - H % stride) % stride
    pad_w = (stride - W % stride) % stride
    
    padded = torch.nn.functional.pad(image_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, H_pad, W_pad = padded.shape
    
    # Output buffer and weight buffer for blending
    output = torch.zeros_like(padded)
    weights = torch.zeros((1, 1, H_pad, W_pad), device=device)
    
    # Create feather mask with min_weight > 0 to avoid division by zero
    def create_feather_mask(h, w, ol, dev):
        mask = torch.ones((1, 1, h, w), device=dev)
        if ol <= 0:
            return mask
        # ramp from 0.01 to 1.0 so edges never have zero weight
        ramp = torch.linspace(0.01, 1.0, ol, device=dev)
        # Top
        mask[:, :, :ol, :] *= ramp.view(-1, 1)
        # Bottom
        mask[:, :, -ol:, :] *= ramp.flip(0).view(-1, 1)
        # Left
        mask[:, :, :, :ol] *= ramp.view(1, -1)
        # Right
        mask[:, :, :, -ol:] *= ramp.flip(0).view(1, -1)
        return mask
    
    # Process tiles
    for y in range(0, H_pad, stride):
        for x in range(0, W_pad, stride):
            y_end = min(y + tile_size, H_pad)
            x_end = min(x + tile_size, W_pad)
            
            tile = padded[:, :, y:y_end, x:x_end]
            
            # Pad tile to multiple of 8 if needed
            _, _, th, tw = tile.shape
            pad_th = (8 - th % 8) % 8
            pad_tw = (8 - tw % 8) % 8
            if pad_th > 0 or pad_tw > 0:
                tile = torch.nn.functional.pad(tile, (0, pad_tw, 0, pad_th), mode='reflect')
            
            # Process tile
            with torch.no_grad():
                I_tile, _, _, _, _ = model(tile)
                I_tile = torch.clamp(I_tile, 0, 1)
            
            # Crop padding
            I_tile = I_tile[:, :, :th, :tw]
            
            # Feather and accumulate
            feather = create_feather_mask(th, tw, min(overlap, th, tw), device)
            output[:, :, y:y_end, x:x_end] += I_tile * feather
            weights[:, :, y:y_end, x:x_end] += feather
    
    # Normalize and crop back
    output = output / weights
    output = output[:, :, :H, :W]
    
    return output


def test_model(model, dataloader, device, save_dir=None, visualize_fields=False, tile_size=None, overlap=256):
    """Test model on dataset.
    
    Args:
        tile_size: If set, use tile-based inference for images larger than this.
                   Recommended: 2048 for 24GB GPU, 2560 for 32GB+ GPU.
        overlap: Overlap between tiles (default 256).
    """
    model.eval()
    
    psnr_list = []
    ssim_list = []
    time_list = []
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if visualize_fields:
            os.makedirs(os.path.join(save_dir, 'fields'), exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            I_low = batch['low'].to(device)
            I_gt = batch['gt'].to(device)
            filenames = batch['filename']
            
            B, C, H, W = I_low.shape
            
            # Measure inference time
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()
            
            # Use tile-based inference for large images
            if tile_size is not None and (H > tile_size or W > tile_size):
                I_out = enhance_image_tiled(model, I_low, device, tile_size=tile_size, overlap=overlap)
                # For tile-based mode, we don't extract fields to save memory
                v = n = c = None
            else:
                # Pad to multiple of 8 for model compatibility
                pad_h = (8 - H % 8) % 8
                pad_w = (8 - W % 8) % 8
                if pad_h > 0 or pad_w > 0:
                    I_low_padded = torch.nn.functional.pad(I_low, (0, pad_w, 0, pad_h), mode='reflect')
                    I_gt_padded = torch.nn.functional.pad(I_gt, (0, pad_w, 0, pad_h), mode='reflect')
                else:
                    I_low_padded = I_low
                    I_gt_padded = I_gt
                
                # Forward pass
                I_out, v, n, c, trajectory = model(I_low_padded)
                
                # Crop back to original size
                if pad_h > 0 or pad_w > 0:
                    I_out = I_out[..., :H, :W]
                    v = v[..., :H//8, :W//8]
                    n = n[..., :H//8, :W//8]
                    c = c[..., :H//8, :W//8]
                    I_low = I_low_padded[..., :H, :W]
                    I_gt = I_gt_padded[..., :H, :W]
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
            
            # Clamp to [0, 1]
            I_out = torch.clamp(I_out, 0, 1)
            
            # Calculate metrics
            for i in range(I_out.shape[0]):
                psnr = calculate_psnr(I_out[i], I_gt[i])
                ssim = calculate_ssim(I_out[i], I_gt[i])
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                
                inference_time = (end_time - start_time) / I_out.shape[0]
                time_list.append(inference_time)
                
                print(f"[{batch_idx * dataloader.batch_size + i + 1}/{len(dataloader.dataset)}] "
                      f"{filenames[i]} - PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, "
                      f"Time: {inference_time * 1000:.2f}ms")
                
                # Save results
                if save_dir is not None:
                    # Save enhanced image
                    save_path = os.path.join(save_dir, filenames[i])
                    save_image(I_out[i], save_path)
                    
                    # Save comparison (low | enhanced | gt)
                    if I_gt is not None:
                        comparison = torch.cat([I_low[i], I_out[i], I_gt[i]], dim=2)
                        comparison_path = os.path.join(save_dir, filenames[i].replace('.', '_compare.'))
                        save_image(comparison, comparison_path)
                    
                    # Visualize fields
                    if visualize_fields and v is not None:
                        # Velocity field
                        v_vis = v[i:i+1].repeat(1, 3, 1, 1)
                        v_vis = torch.nn.functional.interpolate(v_vis, size=I_low.shape[2:], mode='bilinear')
                        v_path = os.path.join(save_dir, 'fields', filenames[i].replace('.', '_velocity.'))
                        save_image(v_vis[0], v_path)
                        
                        # Noise field
                        n_vis = n[i:i+1].repeat(1, 3, 1, 1)
                        n_vis = torch.nn.functional.interpolate(n_vis, size=I_low.shape[2:], mode='bilinear')
                        n_path = os.path.join(save_dir, 'fields', filenames[i].replace('.', '_noise.'))
                        save_image(n_vis[0], n_path)
                        
                        # Color field
                        c_vis = c[i:i+1].repeat(1, 3, 1, 1)
                        c_vis = torch.nn.functional.interpolate(c_vis, size=I_low.shape[2:], mode='bilinear')
                        c_path = os.path.join(save_dir, 'fields', filenames[i].replace('.', '_color.'))
                        save_image(c_vis[0], c_path)
            
            # Free GPU memory after each batch
            del I_low, I_gt, I_out
            if 'I_low_padded' in locals():
                del I_low_padded
            if 'I_gt_padded' in locals():
                del I_gt_padded
            if v is not None:
                del v, n, c, trajectory
            torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    # Print summary
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_time = np.mean(time_list) * 1000  # Convert to ms
    
    print(f"\n{'='*60}")
    print(f"Test Results:")
    print(f"{'='*60}")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average Time: {avg_time:.2f}ms")
    print(f"{'='*60}")
    
    return avg_psnr, avg_ssim, avg_time


def test_single_image(model, image_path, device, save_path=None, img_size=None):
    """
    Test on a single image.
    
    Args:
        img_size: If None, use original resolution (for high-res testing).
                 If > 0, resize to img_size (for training-time resolution).
    """
    model.eval()
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    original_size = img.size  # (W, H)
    
    # Resize only if img_size is specified (training mode)
    if img_size is not None and img_size > 0:
        img = img.resize((img_size, img_size), Image.BICUBIC)
    
    # To tensor
    from torchvision import transforms
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    
    # Pad to multiple of 8 for model compatibility
    _, _, H, W = img_tensor.shape
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h > 0 or pad_w > 0:
        img_tensor_padded = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    else:
        img_tensor_padded = img_tensor
    
    with torch.no_grad():
        # Forward pass
        I_out, v, n, c, trajectory = model(img_tensor_padded)
        I_out = torch.clamp(I_out, 0, 1)
        # Crop back to original size
        if pad_h > 0 or pad_w > 0:
            I_out = I_out[..., :H, :W]
    
    # If testing at original resolution, resize back to original size
    if img_size is None:
        I_out_pil = transforms.ToPILImage()(I_out[0].cpu())
        I_out_pil = I_out_pil.resize(original_size, Image.BICUBIC)
        I_out = transforms.ToTensor()(I_out_pil).unsqueeze(0).to(device)
    
    # Save result
    if save_path is not None:
        save_image(I_out[0], save_path)
        print(f"Saved result to {save_path} (size: {original_size})")
    
    return I_out[0]


def main():
    parser = argparse.ArgumentParser(description='Test SEF-Net v4')
    parser.add_argument('--config', type=str, default='configs/sef_net_v4.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--save_dir', type=str, default='results/sef_net_v4',
                        help='Directory to save results')
    parser.add_argument('--visualize_fields', action='store_true',
                        help='Visualize spatial fields')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--single_image', type=str, default=None,
                        help='Path to single image for testing (uses original resolution)')
    parser.add_argument('--img_size', type=int, default=None,
                        help='Test resolution. None=original (for high-res), >0=resize (for fast test)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for testing')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split: train or test (for LOLv1)')
    parser.add_argument('--no_resize', action='store_true',
                        help='Test at original image resolution (forces batch_size=1)')
    parser.add_argument('--tile_size', type=int, default=2048,
                        help='Tile size for large image inference (0=disable tile-based)')
    parser.add_argument('--overlap', type=int, default=256,
                        help='Overlap between tiles for large image inference')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = SEFNet(
        in_ch=config['model']['in_ch'],
        T=config['model']['T'],
        C_emb=config['model']['C_emb'],
        feat_ch=config['model']['feat_ch'],
        ode_steps=config['model']['ode_steps'],
        epsilon=config['model']['epsilon']
    ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Test single image
    if args.single_image is not None:
        print(f"Testing single image: {args.single_image}")
        save_path = os.path.join(args.save_dir, os.path.basename(args.single_image))
        # Use specified resolution (None=original, >0=resize)
        test_single_image(model, args.single_image, device, save_path, img_size=args.img_size)
        return
    
    # Determine test resolution
    if args.no_resize:
        test_img_size = None
        if args.batch_size != 1:
            print("Warning: original size testing requires batch_size=1, setting batch_size=1")
            args.batch_size = 1
    else:
        test_img_size = args.img_size if args.img_size is not None else config['dataset'].get('img_size', 256)
    
    # Test on dataset
    test_dataset = get_dataset(config, split=args.split, img_size=test_img_size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Determine tile size
    tile_size = args.tile_size if args.tile_size > 0 else None
    
    # Run test
    avg_psnr, avg_ssim, avg_time = test_model(
        model, test_loader, device,
        save_dir=args.save_dir,
        visualize_fields=args.visualize_fields,
        tile_size=tile_size,
        overlap=args.overlap
    )
    
    # Save results to file
    result_file = os.path.join(args.save_dir, 'test_results.txt')
    with open(result_file, 'w') as f:
        f.write(f"SEF-Net v4 Test Results\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: {config['dataset']['name']}\n")
        f.write(f"Average PSNR: {avg_psnr:.4f}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write(f"Average Time: {avg_time:.2f}ms\n")
    
    print(f"Results saved to {result_file}")


if __name__ == '__main__':
    main()
