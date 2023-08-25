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

# Training config
DEVICE = 'cuda' if torch.cuda.is_available() else'cpu'
BATCH_SIZE_TRAIN = 8
BATCH_SIZE_VAL = 9
EPOCHS = 22
NUM_CLASSES = 8
LEARNING_RATE = 0.00001

# Loader config
TRAIN_IMG_DIR = r'./data/cityscapes_data/train'
VAL_IMG_DIR = r'./data/cityscapes_data/val'
PIN_MEMORY = False

# Model config
CHECKPOINT_PATH = r"./unet_model.pth.tar"

INPUT_TRANSFORMS = transforms.Compose([
			transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
			])
	
INPUT_VAL_TRANSFORMS = transforms.Compose([
			transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
			])

INPUT_VAL_TRANSFORMS_INVERSE = transforms.Compose([
			transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
			])
     
def get_saved_model():
    # x classes = x channels
    model = create_model()

    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()
    
    return model

def create_model():
    return UNETScapes(in_channels=3, out_channels=NUM_CLASSES, features=[64, 128, 256, 512]).to(DEVICE)

def train(model, optimizer, loss_fn, scaler, epochs, train_loader):

	gc.collect()
	torch.cuda.empty_cache()

	step_losses = []
	epoch_losses = []

	for epoch in tqdm(range(epochs)):
		epoch_loss = 0

		for X, Y in tqdm(train_loader, total=len(train_loader), leave=False):
			X, Y = X.to(DEVICE), Y.to(DEVICE).long()
			optimizer.zero_grad()

			with torch.cuda.amp.autocast():
				predictions = model(X)
				loss = loss_fn(predictions, Y)

			loss.backward()
			optimizer.step()

			torch.cuda.empty_cache()

			epoch_loss += loss.item()
			step_losses.append(loss.item())

		print(f"Epoch loss: {epoch_loss}\n")
		print(f"Avg epoch loss: {epoch_loss / len(train_loader)}\n")
		epoch_losses.append(epoch_loss / len(train_loader))

	fig, axes = plt.subplots(1, 2, figsize=(10, 5))
	axes[0].plot(step_losses)
	axes[1].plot(epoch_losses)

def main():

	model = create_model()

	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
	loss_fn = nn.CrossEntropyLoss()
	scaler = torch.cuda.amp.GradScaler()

	train_loader, val_loader = get_loaders(
		TRAIN_IMG_DIR,
		VAL_IMG_DIR,
		BATCH_SIZE_TRAIN,
		BATCH_SIZE_VAL,
		INPUT_TRANSFORMS,
		INPUT_VAL_TRANSFORMS,
		PIN_MEMORY,
	)
	
	train(model, optimizer, loss_fn, scaler, EPOCHS, train_loader)

	torch.save(model.state_dict(), CHECKPOINT_PATH)

	show_val_batch_predictions(model, val_loader, INPUT_VAL_TRANSFORMS_INVERSE, 4, DEVICE)

	input()

if __name__ == '__main__':
	main()