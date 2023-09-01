import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from NetModel import UNETScapes
from CityScapesDataset  import get_loaders
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision import transforms
import gc
import os.path
from PIL import Image
from Utils import show_val_batch_predictions
import torch.onnx
import cv2
from Trainer import DEVICE, get_saved_model, INPUT_VAL_TRANSFORMS, INPUT_VAL_TRANSFORMS_INVERSE 

# Evaluate config
EVALUATE_IMG_DIR = r'./data/val'

def evaluate(model, imagesDirPath, input_transforms, input_transforms_inverse):

	X = [Image.open(os.path.join(imagesDirPath, imagePath))for imagePath in os.listdir(imagesDirPath)]
	X = [img.resize((256, 256)) for img in X]
	X = torch.cat(
		[input_transforms(torch.Tensor(np.array(img) / 255 ).permute(2, 0, 1)).unsqueeze(0) for img in X], 0)

	batchSize = X.__len__()		
	X = X.to(DEVICE)

	Y_pred = model(X)
	# output -> [batchsize, classes, x, y]
	# argmax -> take pixel from best match among classes output
	Y_pred = torch.argmax(Y_pred, dim=1)

	fig, axes = plt.subplots(batchSize, 2, figsize=(10, 10))

	for i in range(batchSize):
		landscape = input_transforms_inverse(X[i]).permute(1, 2, 0).cpu().detach().numpy()
		label_class_predicted = Y_pred[i].cpu().detach().numpy()

		axes[i, 0].imshow(landscape)
		axes[i, 0].set_title("Landscape")

		axes[i, 1].imshow(label_class_predicted)
		axes[i, 1].set_title("Label Class - Predicted")
	plt.show()
 
def main():
    model = get_saved_model()
    
    evaluate(model, EVALUATE_IMG_DIR, INPUT_VAL_TRANSFORMS, INPUT_VAL_TRANSFORMS_INVERSE)
    
    input()
    
if __name__ == '__main__':
	main()
    
    