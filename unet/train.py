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
	check_accuracy,
	)
from cityScapesDataset import get_loaders

learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else'cpu'
batch_size = 32
num_epochs = 1
num_workers = 2
image_height = 160
image_width = 160

pin_memory = True
load_model = False

train_img_dir = '../data/cityscapes_data/train'
val_img_dir = '../data/cityscapes_data/val'

def train_fn(loader,model,optimizer,loss_fn,scaler): # runs 1 epoch of training
	loop = tqdm(loader) # progress bar

	for batch_idx,(img,targets) in enumerate(loop):

		optimizer.zero_grad()

		img = img.to(device)
		label = targets.permute(0, 3, 1, 2).to(device)

		output = model(img)
		loss = loss_fn(output, label)

		loss.backward()
		optimizer.step()

		#tqdmloop
		loop.set_postfix(loss=loss.item())

def main():
	train_transform = A.Compose(
		[

			A.Resize(height=image_height, width=image_width),
			ToTensorV2(), ])

	val_transforms = A.Compose(
		[
			A.Resize(height=image_height, width=image_width),
			ToTensorV2(), ])

	model = UNETScapes(in_channels=3, out_channels=3, features= [64,128,256,512]).to(device)

	loss_fn = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	train_loader, val_loader = get_loaders(
		train_img_dir,
		val_img_dir,
		batch_size,
		train_transform,
		val_transforms,
		num_workers,
		pin_memory,
	)
	scaler = torch.cuda.amp.GradScaler()
	for epoch in range(num_workers):
		train_fn(train_loader, model, optimizer, loss_fn, scaler)

		# save
		# check accuracy
		checkpoint = {
			'state_dic': model.state_dict(),
			'optimizer': optimizer.state_dict(),
		}
		save_checkpoint(checkpoint)

		check_accuracy(val_loader, model, device=device)

		save_predictions_as_imgs(
			val_loader, model, folder='saved_images/', device=device)

if __name__ == '__main__':
	main()