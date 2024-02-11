import base64
import io
import time
from PIL import Image
import torch;
import numpy as np;
from torchvision import transforms
import onnxruntime as ort
from matplotlib import pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else'cpu'
INPUT_TRANSFORMS = transforms.Compose([
			transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
			])

def base64ToImage(base64Image : str) -> Image:
    return Image.open(
        io.BytesIO(
            base64.decodebytes(bytes(base64Image, 'utf-8'))
        )
    )
    
def imageToTensor(image : Image) -> torch.Tensor:
    return torch.Tensor((np.array(image) / 255.)).permute(2, 0, 1)   

ort_sess = ort.InferenceSession(r"C:\Users\sikor\ground\ML_playground\Application\Solution\ApiProcesser\segmentor.onnx")


image = Image.open(r"C:\Users\sikor\Desktop\imgs\example_jpg.jpg").resize((256, 256))

start = time.time()

input = INPUT_TRANSFORMS(imageToTensor(image)).numpy()

outputs = ort_sess.run(None, {'input.1': input[np.newaxis, ...]})
result = outputs[0].argmax(axis=1)

end = time.time() - start

print(end)

