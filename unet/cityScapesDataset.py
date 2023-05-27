import os.path

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import torch
import torchvision
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class CityScapesDataset(Dataset):
    def __init__(self, imagesDir, transform=None):
        self.image_dir = imagesDir
        self.transform = transform
        self.images = os.listdir(imagesDir)

        self.num_classes = 10
        self.label_model = KMeans(n_clusters=self.num_classes)
        self.color_array = np.random.choice(range(256), 3*256).reshape(-1, 3)
        self.label_model.fit(self.color_array)

    def __len__(self):
        return len(self.images)

    def split_image_pairs(self, imagePairPath):
        image = np.array(Image.open(imagePairPath).convert('RGB'), dtype = float)
        cityscape, label = image[:, :256, :], image[:, 256:, :]
        return cityscape, label

    def __getitem__(self, index):
        imgPath = os.path.join(self.image_dir, self.images[index])
        image, mask = self.split_image_pairs(imgPath)

        #if self.transform is not None:
            #augmentations = self.transform(image=image, mask=mask)
            #image = augmentations["image"]
            #mask = augmentations["mask"]

        # mask = mask.numpy()
        # mask = np.transpose(mask, (2, 0, 1))

        mask_classes = self.label_model.predict(mask.reshape(-1, 3)) # flatten 3D into 2D & pass to prediction model
        # Clustering pixel values mask 2D into classes values 2D
        mask_classes = mask_classes.reshape(256, 256) # prediction model into original shape 2D

        #fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        #axes[0].imshow(image)
        #axes[1].imshow(mask)
        #axes[2].imshow(mask_classes) # Show 2D with values as classes
        #plt.show()

        mask_classes  = torch.Tensor(mask_classes).long()

        return torch.Tensor(np.transpose(image, (2, 0, 1))), mask_classes

def get_loaders(
    train_dir,
    val_dir,
    batch_size_train,
    batch_size_val,
    train_transform,
    val_transform,
    pin_memory=True,
):
    train_ds = CityScapesDataset(
        imagesDir=train_dir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_train,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CityScapesDataset(
        imagesDir=train_dir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size_val,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader