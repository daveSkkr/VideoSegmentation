import os.path

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import torch
import torchvision

class CityScapesDataset(Dataset):
    def __init__(self, imagesDir, transform=None):
        self.image_dir = imagesDir
        self.transform = transform
        self.images = os.listdir(imagesDir)

    def __len__(self):
        return len(self.images)

    def split_image_pairs(self, imagePairPath):
        image = np.array(Image.open(imagePairPath).convert('RGB'), dtype=np.float32)
        cityscape, label = image[:, :256, :], image[:, 256:, :]
        return cityscape, label

    def __getitem__(self, index):
        imgPath = os.path.join(self.image_dir, self.images[index])
        image, mask = self.split_image_pairs(imgPath)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

def get_loaders(
    train_dir,
    val_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = CityScapesDataset(
        imagesDir=train_dir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CityScapesDataset(
        imagesDir=train_dir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader