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

learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else'cpu'
batch_size = 32
num_workers = 12

pin_memory = True
load_model = False
LEARNING_RATE = 0.0005

train_img_dir = '../data/cityscapes_data/train'
val_img_dir = '../data/cityscapes_data/val'

num_items = 1000
num_classes = 10
color_array = np.random.choice(range(256), 3*num_items).reshape(-1, 3)

label_model = KMeans(n_clusters=num_classes)
label_model.fit(color_array)

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
	loss_fn = nn.CrossEntropyLoss(ignore_index=255)

	train_loader, val_loader = get_loaders(
		train_img_dir,
		val_img_dir,
		batch_size,
		train_transform,
		val_transforms,
		pin_memory,
	)

	scaler = torch.cuda.amp.GradScaler()
	# train
	for epoch in tqdm(range(12)):
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

		# save
		# check accuracy
		checkpoint = {
			'state_dic': model.state_dict(),
			'optimizer': optimizer.state_dict(),
		}

		# save_checkpoint(checkpoint)

		# check_accuracy(val_loader, model, device=device)

		# save predictions
		X, Y = next(iter(val_loader))
		X, Y = X.to(device), Y.to(device)
		Y_pred = model(X)
		print(Y_pred.shape)
		Y_pred = torch.argmax(Y_pred, dim=1)
		print(Y_pred.shape)

		fig, axes = plt.subplots(test_batch_size, 3, figsize=(3 * 5, test_batch_size * 5))

		for i in range(test_batch_size):
			landscape = X[i].permute(1, 2, 0).cpu().detach().numpy()
			label_class = Y[i].cpu().detach().numpy()
			label_class_predicted = Y_pred[i].cpu().detach().numpy()

			axes[i, 0].imshow(landscape)
			axes[i, 0].set_title("Landscape")
			axes[i, 1].imshow(label_class)
			axes[i, 1].set_title("Label Class")
			axes[i, 2].imshow(label_class_predicted)
			axes[i, 2].set_title("Label Class - Predicted")

if __name__ == '__main__':
	main()