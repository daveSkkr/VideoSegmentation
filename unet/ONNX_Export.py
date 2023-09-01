import torch.onnx
from Trainer import get_saved_model

def export_to_onnx():
    
    model = get_saved_model()
    
    model.cpu()
    model.eval()
    
    # Input to the model
    x = torch.randn(1, 3, 256, 256).float()
    torch_out = model(x)

	# Export the model
    torch.onnx.export(model, x, "segmentor.onnx", verbose=True, export_params=True, opset_version=11)
 
    return

if __name__ == '__main__':
    export_to_onnx()