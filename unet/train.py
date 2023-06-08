import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from models import UNETScapes
from utils import(
	load_checkpoint,
	save_checkpoint,
	save_predictions_as_imgs,
	)
from cityScapesDataset import get_loaders
import numpy as np
from labelingUtils import labels
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from torchvision import transforms
import gc

device = 'cuda' if torch.cuda.is_available() else'cpu'
batch_size_train = 32
batch_size_val = 3
epochs = 3

pin_memory = False
LEARNING_RATE = 0.0005

train_img_dir = '../data/cityscapes_data/train'
val_img_dir = '../data/cityscapes_data/val'
checkpoint_path = r"./my_checkpoint.pth.tar"

num_items = 1000
num_classes = 35
color_array = np.random.choice(range(256), 3*num_items).reshape(-1, 3)

label_model = KMeans(n_clusters=num_classes)
label_model.fit(color_array)

def train(model, optimizer, loss_fn, scaler, epochs, train_loader):

	gc.collect()
	torch.cuda.empty_cache()

	step_losses = []
	epoch_losses = []

	for epoch in tqdm(range(epochs)):
		epoch_loss = 0

		for X, Y in tqdm(train_loader, total=len(train_loader), leave=False):
			X, Y = X.to(device), Y.to(device)
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

		torch.save(model.state_dict(), checkpoint_path)

	fig, axes = plt.subplots(1, 2, figsize=(10, 5))
	axes[0].plot(step_losses)
	axes[1].plot(epoch_losses)

def show_predictions_plot(model, val_loader, inverse_transforms):

	X, Y = next(iter(val_loader))
	X, Y = X.to(device), Y.to(device)
	Y_pred = model(X)
	print(Y_pred.shape)
	Y_pred = torch.argmax(Y_pred, dim=1)
	print(Y_pred.shape)
	fig, axes = plt.subplots(batch_size_val, 3, figsize=(3 * 5, 5 * 5))

	for i in range(batch_size_val):
		landscape = inverse_transforms(X[i]).permute(1, 2, 0).cpu().detach().long().numpy()
		label_class = Y[i].cpu().detach().numpy()
		label_class_predicted = Y_pred[i].cpu().detach().numpy()

		axes[i, 0].imshow(landscape)
		axes[i, 0].set_title("Landscape")
		axes[i, 1].imshow(label_class)
		axes[i, 1].set_title("Label Class")
		axes[i, 2].imshow(label_class_predicted)
		axes[i, 2].set_title("Label Class - Predicted")
	plt.show()

def main():

	idTocolor = {label.id: np.asarray(label.color) for label in labels}

	input_transforms = transforms.Compose([
			#transforms.ToTensor(),
			#transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
		])

	input_transforms_inverse = transforms.Compose([
		#transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
	])

	# 10 classes = 10 channels
	model = UNETScapes(in_channels=3, out_channels=num_classes, features= [64,128,256,512]).to(device)

	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
	loss_fn = nn.CrossEntropyLoss()
	scaler = torch.cuda.amp.GradScaler()

	train_loader, val_loader = get_loaders(
		train_img_dir,
		val_img_dir,
		batch_size_train,
		batch_size_val,
		input_transforms,
		pin_memory,
	)

	train(model, optimizer, loss_fn, scaler, epochs, train_loader)

	#model.load_state_dict(torch.load(checkpoint_path))
	#model.eval()

	# check predictions for batch
	show_predictions_plot(model, val_loader, input_transforms_inverse)

	input()


if __name__ == '__main__':
	main()