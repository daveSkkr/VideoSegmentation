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
from labelingUtils import trainId2label as t2l
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else'cpu'
batch_size_train = 32
batch_size_val= 8
num_workers = 12

pin_memory = True
load_model = False
LEARNING_RATE = 0.0005

train_img_dir = '../data/cityscapes_data/train'
val_img_dir = '../data/cityscapes_data/val'
checkpoint_path = r"./my_checkpoint.pth.tar"

num_items = 1000
num_classes = 10
color_array = np.random.choice(range(256), 3*num_items).reshape(-1, 3)

label_model = KMeans(n_clusters=num_classes)
label_model.fit(color_array)


def train(model, optimizer, loss_fn, scaler, epochs, train_loader):
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
			epoch_loss += loss.item()
			step_losses.append(loss.item())

		epoch_losses.append(epoch_loss)

		torch.save(model.state_dict(), checkpoint_path)

def main():
	train_transform = A.Compose(
		[
			ToTensorV2(),
		])

	val_transforms = A.Compose(
		[
			ToTensorV2(),
		])

	# 19 classes = 19 channels
	model = UNETScapes(in_channels=3, out_channels=19, features= [64,128,256,512]).to(device)

	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
	loss_fn = nn.CrossEntropyLoss()
	scaler = torch.cuda.amp.GradScaler()

	train_loader, val_loader = get_loaders(
		train_img_dir,
		val_img_dir,
		batch_size_train,
		batch_size_val,
		train_transform,
		val_transforms,
		pin_memory,
	)

	#train(model, optimizer, loss_fn, scaler, 100, train_loader)

	model.load_state_dict(torch.load(checkpoint_path))
	model.eval()

	# check predictions for batch
	X, Y = next(iter(val_loader))
	X, Y = X.to(device), Y.to(device)
	Y_pred = model(X)
	print(Y_pred.shape)
	Y_pred = torch.argmax(Y_pred, dim=1)
	print(Y_pred.shape)

	fig, axes = plt.subplots(batch_size_val, 3, figsize=(3 * 5, batch_size_val * 5))

	for i in range(batch_size_val):
		landscape = X[i].permute(1, 2, 0).cpu().detach().numpy()
		label_class = Y[i].cpu().detach().numpy()
		label_class_predicted = Y_pred[i].cpu().detach().numpy()

		axes[i, 0].imshow(landscape)
		axes[i, 0].set_title("Landscape")
		axes[i, 1].imshow(label_class)
		axes[i, 1].set_title("Label Class")
		axes[i, 2].imshow(label_class_predicted)
		axes[i, 2].set_title("Label Class - Predicted")

	plt.show()

	input()

if __name__ == '__main__':
	main()