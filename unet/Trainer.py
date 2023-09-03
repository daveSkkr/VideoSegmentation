from collections import namedtuple
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from NetModel import UNETScapes
from CityScapesDataset  import get_loaders
import matplotlib.pyplot as plt
from torchvision import transforms
import gc
import os.path
from Utils import show_val_batch_predictions
import torch.onnx
from torchsummary import summary

# Training config
DEVICE = 'cuda' if torch.cuda.is_available() else'cpu'
BATCH_SIZE_TRAIN = 8
BATCH_SIZE_VAL = 9
EPOCHS = 20
NUM_CLASSES = 12
LEARNING_RATE = 0.00005

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

def train(model, optimizer, loss_fn, scaler, epochs, train_loader, val_loader):

	gc.collect()
	torch.cuda.empty_cache()

	# Declaring namedtuple()
	EpochLoss = namedtuple('EpochLoss', 'Train Validation')
	epoch_losses = []

	for epoch in tqdm(range(epochs)):
		train_loss = 0

		# training epoch
		for X, Y in tqdm(train_loader, total=len(train_loader), leave=False):
			X, Y = X.to(DEVICE), Y.to(DEVICE).long()
			optimizer.zero_grad()

			with torch.cuda.amp.autocast():
				predictions = model(X)
				loss = loss_fn(predictions, Y)

			loss.backward()
			optimizer.step()

			torch.cuda.empty_cache()

			train_loss += loss.item()
	
		train_avg_loss = train_loss / len(train_loader)
		print(f"Avg epoch train loss: {train_avg_loss}\n")
   
		# validation set epoch
		validation_loss = 0
		for X, Y in tqdm(val_loader, total = len(val_loader)):
			X, Y = X.to(DEVICE), Y.to(DEVICE).long()
   
			with torch.cuda.amp.autocast():
				predictions = model(X)
				loss = loss_fn(predictions, Y)

			validation_loss += loss.item()

		validation_avg_loss = validation_loss / len(val_loader)
		print(f"Avg validation set loss: {validation_avg_loss}\n")
  
		epoch_losses.append( 
			EpochLoss(
      			train_avg_loss, 
				validation_avg_loss
         		)
		)

	fig, axes = plt.subplots(1, 2, figsize=(10, 5))
	axes[0].plot([tl.Train for tl in epoch_losses])
	axes[1].plot([vl.Validation for vl in epoch_losses])
 
	plt.show()

def main():

	model = create_model()

	# summary(model, input_size=(3, 256, 256), batch_size=3, device=DEVICE )

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
	
	train(model, optimizer, loss_fn, scaler, EPOCHS, train_loader, val_loader)

	torch.save(model.state_dict(), CHECKPOINT_PATH)

	show_val_batch_predictions(model, val_loader, INPUT_VAL_TRANSFORMS_INVERSE, 4, DEVICE)

	input()

if __name__ == '__main__':
	main()