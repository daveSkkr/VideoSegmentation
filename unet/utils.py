import torch
import torchvision
from cityScapesDataset import CityScapesDataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else'cpu'
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def save_predictions_as_imgs(
    loader, model, inverse_transform, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):

        x, y = x.to(device), y.to(device)
        predictions = model(x)

        predictions = torch.nn.functional.softmax(predictions, dim=1)
        pred_labels = torch.argmax(predictions, dim=1)
        pred_labels = pred_labels.float()

        # Remapping the labels
        pred_labels = pred_labels.to('cpu')
        pred_labels.apply_(lambda x: trainId2label[x].id)
        pred_labels = pred_labels.to(device)

        # Resizing predicted images too original size
        pred_labels = transforms.Resize((1024, 2048))(pred_labels)

        # Configure filename & location to save predictions as images
        s = str(s)
        pos = s.rfind('/', 0, len(s))
        name = s[pos + 1:-18]


        global location
        location = 'saved_images\multiclass_1'

        utils.save_as_images(pred_labels, location, name, multiclass=True)