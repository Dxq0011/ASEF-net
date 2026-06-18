"""
Dataset loaders for ASEF-Net
Supports LOL-v1, LOL-v2, SID, and custom datasets
"""
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import random


class LowLightDataset(Dataset):
    """Paired low-light / normal-light dataset"""
    def __init__(self, low_dir, gt_dir, img_size=256, augment=True):
        self.low_dir = low_dir
        self.gt_dir = gt_dir
        self.img_size = img_size
        self.augment = augment

        # Get file lists
        self.low_files = sorted(glob.glob(os.path.join(low_dir, '*.*')))
        self.gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.*')))

        assert len(self.low_files) == len(self.gt_files), \
            f"Mismatch: {len(self.low_files)} low vs {len(self.gt_files)} gt"

        # Supported extensions
        self.valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

        self.low_files = [f for f in self.low_files if f.lower().endswith(self.valid_ext)]
        self.gt_files = [f for f in self.gt_files if f.lower().endswith(self.valid_ext)]

    def __len__(self):
        return len(self.low_files)

    def _random_crop(self, img_low, img_gt):
        """Random crop to img_size, applied synchronously."""
        w, h = img_low.size
        th, tw = self.img_size, self.img_size

        if w < tw or h < th:
            # Scale up if image is smaller than crop size
            scale = max(tw / w, th / h) * 1.2
            new_w, new_h = int(w * scale), int(h * scale)
            img_low = img_low.resize((new_w, new_h), Image.BICUBIC)
            img_gt = img_gt.resize((new_w, new_h), Image.BICUBIC)
            w, h = new_w, new_h

        if w == tw and h == th:
            return img_low, img_gt

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        img_low = img_low.crop((j, i, j + tw, i + th))
        img_gt = img_gt.crop((j, i, j + tw, i + th))
        return img_low, img_gt

    def _apply_color_jitter(self, low_tensor, gt_tensor):
        """Apply same color jitter to both low and gt."""
        # Brightness
        brightness_factor = random.uniform(0.7, 1.3)
        low_tensor = transforms.functional.adjust_brightness(low_tensor, brightness_factor)
        gt_tensor = transforms.functional.adjust_brightness(gt_tensor, brightness_factor)

        # Contrast
        contrast_factor = random.uniform(0.8, 1.2)
        low_tensor = transforms.functional.adjust_contrast(low_tensor, contrast_factor)
        gt_tensor = transforms.functional.adjust_contrast(gt_tensor, contrast_factor)

        # Saturation
        saturation_factor = random.uniform(0.8, 1.2)
        low_tensor = transforms.functional.adjust_saturation(low_tensor, saturation_factor)
        gt_tensor = transforms.functional.adjust_saturation(gt_tensor, saturation_factor)

        return low_tensor, gt_tensor

    def _apply_gamma(self, low_tensor, gt_tensor):
        """Apply random gamma to simulate different exposures."""
        gamma = random.uniform(0.7, 1.3)
        # Apply stronger gamma variation to low image
        low_tensor = transforms.functional.adjust_gamma(low_tensor, gamma)
        # Apply milder gamma to gt to keep relative consistency
        gt_gamma = random.uniform(0.9, 1.1)
        gt_tensor = transforms.functional.adjust_gamma(gt_tensor, gt_gamma)
        return low_tensor, gt_tensor

    def __getitem__(self, idx):
        # Load images
        low_img = Image.open(self.low_files[idx]).convert('RGB')
        gt_img = Image.open(self.gt_files[idx]).convert('RGB')

        # Synchronous augmentations on PIL images first
        if self.augment:
            # Random crop (instead of global resize) - key augmentation!
            if self.img_size is not None and self.img_size > 0:
                low_img, gt_img = self._random_crop(low_img, gt_img)

            # Random horizontal flip
            if random.random() > 0.5:
                low_img = low_img.transpose(Image.FLIP_LEFT_RIGHT)
                gt_img = gt_img.transpose(Image.FLIP_LEFT_RIGHT)

            # Random vertical flip
            if random.random() > 0.5:
                low_img = low_img.transpose(Image.FLIP_TOP_BOTTOM)
                gt_img = gt_img.transpose(Image.FLIP_TOP_BOTTOM)

            # Random rotation 90, 180, 270
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                low_img = low_img.rotate(angle, Image.BICUBIC)
                gt_img = gt_img.rotate(angle, Image.BICUBIC)
        else:
            # Test/val: resize to img_size
            if self.img_size is not None and self.img_size > 0:
                low_img = low_img.resize((self.img_size, self.img_size), Image.BICUBIC)
                gt_img = gt_img.resize((self.img_size, self.img_size), Image.BICUBIC)

        # To tensor [0, 1]
        low_tensor = transforms.ToTensor()(low_img)
        gt_tensor = transforms.ToTensor()(gt_img)

        # Color augmentations on tensors (only for training)
        if self.augment:
            low_tensor, gt_tensor = self._apply_color_jitter(low_tensor, gt_tensor)
            low_tensor, gt_tensor = self._apply_gamma(low_tensor, gt_tensor)

        return {
            'low': low_tensor,
            'gt': gt_tensor,
            'filename': os.path.basename(self.low_files[idx])
        }


class LOLv1Dataset(LowLightDataset):
    """LOL-v1 dataset"""
    def __init__(self, root_dir, split='train', img_size=256, augment=True):
        if split == 'train':
            low_dir = os.path.join(root_dir, 'our485', 'low')
            gt_dir = os.path.join(root_dir, 'our485', 'high')
        else:
            low_dir = os.path.join(root_dir, 'eval15', 'low')
            gt_dir = os.path.join(root_dir, 'eval15', 'high')

        super().__init__(low_dir, gt_dir, img_size, augment)


class LOLv2Dataset(LowLightDataset):
    """LOL-v2 dataset (Real or Synthetic)"""
    def __init__(self, root_dir, subset='Real', split='train', img_size=256, augment=True):
        low_dir = os.path.join(root_dir, subset, split, 'low')
        gt_dir = os.path.join(root_dir, subset, split, 'high')
        super().__init__(low_dir, gt_dir, img_size, augment)


class SIDDataset(Dataset):
    """See-in-the-Dark dataset (RAW processing)"""
    def __init__(self, root_dir, split='train', img_size=512):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size

        # SID uses RAW files, here we provide a simplified RGB version
        # For full RAW implementation, see official SID code
        low_dir = os.path.join(root_dir, split, 'low')
        gt_dir = os.path.join(root_dir, split, 'high')

        self.low_files = sorted(glob.glob(os.path.join(low_dir, '*.png')))
        self.gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.png')))

    def __len__(self):
        return len(self.low_files)

    def __getitem__(self, idx):
        low_img = Image.open(self.low_files[idx]).convert('RGB')
        gt_img = Image.open(self.gt_files[idx]).convert('RGB')

        if self.img_size is not None and self.img_size > 0:
            low_img = low_img.resize((self.img_size, self.img_size), Image.BICUBIC)
            gt_img = gt_img.resize((self.img_size, self.img_size), Image.BICUBIC)

        return {
            'low': transforms.ToTensor()(low_img),
            'gt': transforms.ToTensor()(gt_img),
            'filename': os.path.basename(self.low_files[idx])
        }


def get_dataloader(dataset_name, root_dir, split='train', batch_size=8, 
                   img_size=256, num_workers=4, augment=True):
    """Factory function to create dataloaders"""
    if dataset_name == 'lol_v1':
        dataset = LOLv1Dataset(root_dir, split, img_size, augment)
    elif dataset_name == 'lol_v2_real':
        dataset = LOLv2Dataset(root_dir, 'Real', split, img_size, augment)
    elif dataset_name == 'lol_v2_synthetic':
        dataset = LOLv2Dataset(root_dir, 'Synthetic', split, img_size, augment)
    elif dataset_name == 'sid':
        dataset = SIDDataset(root_dir, split, img_size)
    else:
        # Custom dataset
        low_dir = os.path.join(root_dir, split, 'low')
        gt_dir = os.path.join(root_dir, split, 'high')
        dataset = LowLightDataset(low_dir, gt_dir, img_size, augment)

    shuffle = (split == 'train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, pin_memory=True)

    return dataloader
