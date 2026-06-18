"""
Visualization utilities for ASEF-Net
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def visualize_fields(v, n, c, save_path=None):
    """Visualize v, n, c fields as heatmaps"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    fields = [v, n, c]
    titles = ['Exposure Velocity v(x,y)', 'Noise Confidence n(x,y)', 'Color Stability c(x,y)']
    cmaps = ['hot', 'viridis', 'plasma']

    for i, (field, title, cmap) in enumerate(zip(fields, titles, cmaps)):
        if isinstance(field, torch.Tensor):
            field = field.detach().cpu().numpy()
        if field.ndim == 4:
            field = field[0, 0]  # Take first sample
        else:
            field = field[0]

        im = axes[i].imshow(field, cmap=cmap)
        axes[i].set_title(title, fontsize=12)
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_stages(states, save_path=None):
    """Visualize virtual exposure stages"""
    num_stages = len(states)
    cols = min(num_stages, 6)
    rows = (num_stages + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for i, state in enumerate(states):
        row = i // cols
        col = i % cols

        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        img = np.transpose(state[0], (1, 2, 0))  # [H, W, C]
        img = np.clip(img, 0, 1)

        axes[row, col].imshow(img)
        axes[row, col].set_title(f'Stage {i}', fontsize=10)
        axes[row, col].axis('off')

    # Hide unused subplots
    for i in range(num_stages, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_comparison(I_low, I_out, I_gt, save_path=None):
    """Compare low-light input, enhanced output, and ground truth"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    images = [I_low, I_out, I_gt]
    titles = ['Low-Light Input', 'ASEF-Net Output', 'Ground Truth']

    for i, (img, title) in enumerate(zip(images, titles)):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        img = np.transpose(img[0], (1, 2, 0))
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=12)
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_velocity_heatmap(v, I_low, save_path=None):
    """Overlay velocity heatmap on input image"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if isinstance(I_low, torch.Tensor):
        I_low = I_low.detach().cpu().numpy()
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()

    img = np.transpose(I_low[0], (1, 2, 0))
    img = np.clip(img, 0, 1)
    vel = v[0, 0]

    axes[0].imshow(img)
    axes[0].set_title('Input Image', fontsize=12)
    axes[0].axis('off')

    im = axes[1].imshow(vel, cmap='jet', alpha=0.7)
    axes[1].imshow(img, alpha=0.3)
    axes[1].set_title('Exposure Velocity v(x,y)', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
