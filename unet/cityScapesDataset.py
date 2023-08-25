import os.path
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from DatasetLabeling import idx_to_category, idx_to_color

class CityScapesDataset(Dataset):
    
    def __init__(self, imagesDir, transform=None, target_transform =None):
        self.transform = transform
        self.target_transform = target_transform
        self.image_dir = imagesDir
        self.images = os.listdir(imagesDir)

        self.X = []
        self.Y = []

        self.load_images()

    # load dataset into RAM, due to rational size to speed up computation
    def load_images(self):
        for imagePath in tqdm(self.images):
            raw, classes = self.preprocess_image(os.path.join(self.image_dir, imagePath))
            self.X.append(torch.Tensor(raw / 255.).permute(2, 0, 1))
            self.Y.append(torch.Tensor(classes))
            
        # self.visualizeBatch(4)
        
        return
            
    def visualizeBatch(self, batchSize):
        fig, axes = plt.subplots(batchSize, 2, figsize=(4, 2. * batchSize), squeeze=True)
        fig.subplots_adjust(hspace=0.0, wspace=0.0)

        for i in range(batchSize):
            img, mask = self.X[i], self.Y[i]
            # print(img.shape, mask.shape)
            axes[i, 0].imshow(img.permute(1, 2, 0))
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])

            axes[i, 1].imshow(mask, cmap='magma')
            axes[i, 1].set_xticks([])
            axes[i, 1].set_yticks([])

    def preprocess_image(self, imagePath):
        raw, mask = self.split_image_pairs(imagePath)
        height, width, channels = mask.shape
        
        # L2 algorithm
        # compute then the sum of squared distances for each pixel to the colors (L2 between the color and pixel data) :
        # the value which will be the minimal is the category name we will use for that pixel, and we will get it using argmin
        distances = np.sum((mask.reshape(-1, channels)[:, np.newaxis, :] - idx_to_color) ** 2, axis=2)
        classes = np.argmin(distances, axis=1).reshape(height, width)
        
        # narrow classes
        narrowClasses = np.vectorize(lambda idx: idx_to_category[idx])
        classes = narrowClasses(classes)
        
        return (raw, classes)

    def split_image_pairs(self, imagePairPath):
        image = np.array(Image.open(imagePairPath))
        raw, mask = image[:, :256, :], image[:, 256:, :]
        return raw, mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x , y

def get_loaders(
    train_dir,
    val_dir,
    batch_size_train,
    batch_size_val,
    transforms,
    val_transform,
    pin_memory=True,
):
    train_ds = CityScapesDataset(
        imagesDir=train_dir,
        transform=transforms,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_train,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CityScapesDataset(
        imagesDir=val_dir,
        transform=val_transform,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size_train,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader