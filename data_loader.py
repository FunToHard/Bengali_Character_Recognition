import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class BengaliHandwritingDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        """
        Args:
            root_dir (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
            train (bool): Whether this is training or validation dataset
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.samples = []
        self.class_to_idx = {}
        
        # Determine the split directory (train or val)
        split_dir = os.path.join(root_dir, 'train' if train else 'val')
        
        # Get all class directories (each representing a Bengali character)
        class_dirs = [d for d in os.listdir(split_dir) 
                     if os.path.isdir(os.path.join(split_dir, d))]
        class_dirs.sort()  # Ensure consistent ordering
        
        # Create class to index mapping
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_dirs)}
        
        # Collect all image samples
        for class_name in class_dirs:
            class_idx = self.class_to_idx[class_name]
            class_dir = os.path.join(split_dir, class_name)
            
            # Get all image files in the class directory
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = self.samples[idx][0]
        image = Image.open(img_name).convert('L')  # Convert to grayscale
        label = self.samples[idx][1]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_loaders(data_dir, batch_size=32):
    """
    Create training and validation data loaders
    """    # Define transforms with enhanced augmentation for handwritten characters
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomAffine(
            degrees=15,  # More rotation for handwriting variation
            translate=(0.1, 0.1),  # Translation for position invariance
            scale=(0.85, 1.15),  # Scaling for size variation
            shear=(-15, 15)  # Shearing for slant variation
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Perspective changes
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Handling different pen pressures
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create datasets
    train_dataset = BengaliHandwritingDataset(
        root_dir=data_dir,
        transform=train_transform,
        train=True
    )
    
    val_dataset = BengaliHandwritingDataset(
        root_dir=data_dir,
        transform=val_transform,
        train=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader
