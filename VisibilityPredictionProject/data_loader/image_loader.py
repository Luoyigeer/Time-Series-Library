"""
Image Data Loader for Visibility Detection
==========================================

This module implements a data loader for image-based visibility detection.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, List
import warnings
from PIL import Image
import torchvision.transforms as transforms


class VisibilityImageLoader(Dataset):
    """
    Dataset loader for image-based visibility detection.
    
    This loader processes images with corresponding visibility labels,
    useful for single-image visibility estimation and transfer learning.
    
    Args:
        data_path: Path to image directory
        labels_file: Path to CSV file with image labels
        split: Data split - 'train', 'val', or 'test'
        image_size: Size to resize images to (height, width)
        normalize: Whether to normalize images
        augment: Whether to apply data augmentation (train only)
    """
    
    def __init__(
        self,
        data_path: str,
        labels_file: Optional[str] = None,
        split: str = 'train',
        image_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        augment: bool = True
    ):
        super(VisibilityImageLoader, self).__init__()
        
        self.data_path = Path(data_path)
        self.split = split
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment and (split == 'train')
        
        # Load image paths and labels
        self.image_paths, self.labels = self._load_data(labels_file)
        
        # Setup transforms
        self.transform = self._get_transforms()
        
        print(f"Loaded {len(self.image_paths)} images for {split} split")
    
    def _load_data(self, labels_file: Optional[str]) -> Tuple[List[Path], List[float]]:
        """
        Load image paths and corresponding visibility labels.
        
        Returns:
            List of image paths and list of visibility values
        """
        if self.data_path.exists():
            # Load from directory
            image_paths = sorted(list(self.data_path.glob('*.jpg')) + 
                               list(self.data_path.glob('*.png')))
            
            if labels_file is not None and Path(labels_file).exists():
                # Load labels from CSV
                import pandas as pd
                labels_df = pd.read_csv(labels_file)
                
                # Match images to labels
                labels = []
                valid_paths = []
                for img_path in image_paths:
                    img_name = img_path.name
                    if img_name in labels_df['image'].values:
                        label = labels_df[labels_df['image'] == img_name]['visibility'].values[0]
                        labels.append(label)
                        valid_paths.append(img_path)
                
                image_paths = valid_paths
            else:
                # Generate synthetic labels for demonstration
                warnings.warn("No labels file found. Generating synthetic labels.")
                labels = [np.random.uniform(0, 20) for _ in image_paths]
        else:
            # Generate synthetic dataset
            warnings.warn(f"Data path {self.data_path} not found. Creating synthetic dataset.")
            image_paths, labels = self._generate_synthetic_dataset()
        
        # Split data
        n_total = len(image_paths)
        train_end = int(n_total * 0.7)
        val_end = int(n_total * 0.85)
        
        if self.split == 'train':
            image_paths = image_paths[:train_end]
            labels = labels[:train_end]
        elif self.split == 'val':
            image_paths = image_paths[train_end:val_end]
            labels = labels[train_end:val_end]
        else:  # test
            image_paths = image_paths[val_end:]
            labels = labels[val_end:]
        
        return image_paths, labels
    
    def _generate_synthetic_dataset(self) -> Tuple[List[str], List[float]]:
        """Generate synthetic image paths and labels for demonstration."""
        n_samples = 1000
        
        # Create synthetic "paths" (will generate images on the fly)
        image_paths = [f'synthetic_image_{i}.jpg' for i in range(n_samples)]
        
        # Generate synthetic visibility labels
        labels = np.random.uniform(0, 20, n_samples).tolist()
        
        return image_paths, labels
    
    def _get_transforms(self):
        """Get image transforms for preprocessing."""
        transform_list = []
        
        # Resize
        transform_list.append(transforms.Resize(self.image_size))
        
        # Data augmentation for training
        if self.augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalize
        if self.normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        return transforms.Compose(transform_list)
    
    def _load_image(self, image_path) -> Image.Image:
        """Load an image from path or generate synthetic."""
        if isinstance(image_path, Path) and image_path.exists():
            # Load real image
            image = Image.open(image_path).convert('RGB')
        else:
            # Generate synthetic image
            # Create a random image with patterns
            np.random.seed(hash(str(image_path)) % 2**32)
            img_array = np.random.rand(*self.image_size, 3) * 255
            image = Image.fromarray(img_array.astype('uint8'), mode='RGB')
        
        return image
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            image: Image tensor [3, height, width]
            label: Visibility value (scalar)
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and transform image
        image = self._load_image(image_path)
        image = self.transform(image)
        
        return image, torch.FloatTensor([label])


def create_image_dataloader(
    data_path: str,
    batch_size: int = 32,
    split: str = 'train',
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for visibility image data.
    
    Args:
        data_path: Path to image directory
        batch_size: Batch size
        split: Data split
        **kwargs: Additional arguments for VisibilityImageLoader
    
    Returns:
        DataLoader instance
    """
    dataset = VisibilityImageLoader(data_path, split=split, **kwargs)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader
