import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from models import UNETScapes
from cityScapesDataset import get_loaders
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import gc
from cityScapesDataset import CityScapesDataset
from utils import show_predictions_sample_plot
import os.path
from PIL import Image

from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else'cpu'
batch_size_train = 8
batch_size_val = 9
epochs = 20

pin_memory = False
LEARNING_RATE = 1e-4

train_img_dir = '../data/cityscapes_data/train'
val_img_dir = '../data/cityscapes_data/val'
evaluate_img_dir = r'../data/val'
checkpoint_path = r"./my_checkpoint.pth.tar"

num_classes = 35

def train(model, optimizer, loss_fn, scaler, epochs, train_loader):

	gc.collect()
	torch.cuda.empty_cache()

	step_losses = []
	epoch_losses = []

	for epoch in tqdm(range(epochs)):
		epoch_loss = 0

		for X, Y in tqdm(train_loader, total=len(train_loader), leave=False):
			X, Y = X.to(device), Y.to(device).long()
			optimizer.zero_grad()

			with torch.cuda.amp.autocast():
				predictions = model(X)
				loss = loss_fn(predictions, Y)

			loss.backward()
			optimizer.step()

			torch.cuda.empty_cache()

			epoch_loss += loss.item()
			step_losses.append(loss.item())

		epoch_losses.append(epoch_loss)

	fig, axes = plt.subplots(1, 2, figsize=(10, 5))
	axes[0].plot(step_losses)
	axes[1].plot(epoch_losses)

def showImagesToEvaluate():
	imagesToValue = os.listdir(evaluate_img_dir)

	images = [np.array(Image.open(os.path.join(evaluate_img_dir, imagePath))) for imagePath in imagesToValue]

	fig, axes = plt.subplots(images.__len__(), 1, figsize=(10, 4. * images.__len__()), squeeze=True)
	fig.subplots_adjust(hspace=0.0, wspace=0.0)

	for i in range(images.__len__()):
		axes[i].imshow(images[i])
		axes[i].set_xticks([])
		axes[i].set_yticks([])

	plt.show()

def main():
	showImagesToEvaluate()

	input_transforms = transforms.Compose([
			transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
		])

	input_transforms_inverse = transforms.Compose([
		transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
	])

	# x classes = x channels
	model = UNETScapes(in_channels=3, out_channels=num_classes, features= [64,128,256,512]).to(device)

	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
	loss_fn = nn.CrossEntropyLoss()
	scaler = torch.cuda.amp.GradScaler()

	train_loader, val_loader = get_loaders(
		train_img_dir,
		evaluate_img_dir,
		batch_size_train,
		batch_size_val,
		input_transforms,
		pin_memory,
	)

	val_ds = CityScapesDataset(
		imagesDir=evaluate_img_dir,
		transform=None,
	)

	val_loader = DataLoader(
		val_ds,
		batch_size=4,
		pin_memory=pin_memory,
		shuffle=False,
	)

	#model.load_state_dict(torch.load(checkpoint_path))
	#model.eval()

	train(model, optimizer, loss_fn, scaler, epochs, train_loader)

	torch.save(model.state_dict(), checkpoint_path)

	# check predictions for batch
	show_predictions_sample_plot(model, val_loader, input_transforms_inverse, batch_size_val, device)

	input()

if __name__ == '__main__':
	main()