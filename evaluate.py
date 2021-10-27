import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from Unet_model import build_unet
import torch.nn.functional as F

batch_size = 64
model_path = "/media/khanhpham/새 볼륨/unet_kits19_data/checkpoint.pth"
if torch.cuda.is_available():
    device = 'cuda:0'
    print('Running on the GPU')
else:
    device = 'cpu'
    print('Running on the CPU')

def test(data, model):    
    with torch.no_grad():
        for batch_idx, (image,mask)in enumerate(tqdm(data)):
            img = image.to(device)
            mask = mask.to(device)
            predictions = model(img) 
            predictions = F.softmax(predictions, dim=1)
            pred_labels = torch.argmax(predictions, dim=1) 
            pred_labels = pred_labels.float()
def main():
    test_loader = get_loaders(
        get_dir = "/media/khanhpham/새 볼륨/unet_kits19_data/test/",
        batch_size = batch_size,
        img_transform = None,
        data_shuffle=False,
    )
    model = buld_unet(in_c=in_channel,out_c=out_channel)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
