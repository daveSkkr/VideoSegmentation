import os.path

from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CityScapesDataset(Dataset):
    def __init__(self, imagesDir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        imgPath = os.path.join(self.image_dir, self.images[index])
        image, mask = split_image_pairs(imgPath)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


    def split_image_pairs(self, imagePairPath):
        imageMaskPair = Image.open(imagePairPath)
        image, mask = imageMaskPair.crop([0, 0, 256, 256], [256, 0, 512, 256] )

        return np.array(image).convert("RGB"), np.array(mask).convert("RGB")