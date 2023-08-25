import PIL
import cv2
import torch
from torchvision import transforms
from models import UNETScapes
from train import evaluate
from train import get_model
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

evaluate_img_dir = r'../data/val'

input_transforms = transforms.Compose([
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

input_transforms_inverse = transforms.Compose([
    transforms.Normalize((-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225), (1 / 0.229, 1 / 0.224, 1 / 0.225))
])

def main():
    model = get_model()

    vidcap = cv2.VideoCapture(r'..\data\video.mp4')
    success, image = vidcap.read()
    count = 0

    while count < 100:
        success, image = vidcap.read()
        image = processFrame(model, image)

        plt.imsave("frame%d.png" % count, image)

        print('Read a new frame: ', success)
        count += 1

    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (720, 1080))
    for frame in frames:
        out.write(frame)  # frame is a numpy.ndarray with shape (1280, 720, 3)
    out.release()

def processFrame(model, frame):
    frame = Image.fromarray(frame.astype('uint8'), 'RGB')
    frame.thumbnail((256, 256), Image.ANTIALIAS)

    X = torch.cat(
        [input_transforms(
            torch.Tensor(np.array(img) / 255.0).permute(2, 0, 1)).unsqueeze(0) for img in [frame]], 0)

    X = X.to("cuda")

    Y_pred = model(X)
    Y_pred = torch.argmax(Y_pred, dim=1).cpu().detach().numpy()[0]

    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=Y_pred.min(), vmax=Y_pred.max())

    return cmap(norm(Y_pred))

if __name__ == '__main__':
	main()