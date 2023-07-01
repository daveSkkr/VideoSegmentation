import torch
import torchvision
from cityScapesDataset import CityScapesDataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from torch import Tensor
import matplotlib.pyplot as plt

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def show_predictions_sample_plot(model, val_loader, inverse_transforms, batch_size_val, device):

	X, Y = next(iter(val_loader))
	X, Y = X.to(device), Y.to(device)
	Y_pred = model(X)
	print(Y_pred.shape)
	Y_pred = torch.argmax(Y_pred, dim=1)
	print(Y_pred.shape)
	fig, axes = plt.subplots(batch_size_val, 3, figsize=(3 * 5, 5 * 5))

	for i in range(batch_size_val):
		landscape = inverse_transforms(X[i]).permute(1, 2, 0).cpu().detach().numpy()
		label_class = Y[i].cpu().detach().numpy()
		label_class_predicted = Y_pred[i].cpu().detach().numpy()

		axes[i, 0].imshow(landscape)
		axes[i, 0].set_title("Landscape")
		axes[i, 1].imshow(label_class)
		axes[i, 1].set_title("Label Class")
		axes[i, 2].imshow(label_class_predicted)
		axes[i, 2].set_title("Label Class - Predicted")
	plt.show()