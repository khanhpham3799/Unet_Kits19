import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from Unet_model import UNET
from utils import dice_score, get_loaders
import matplotlib.pyplot as plt
import numpy as np
import random
from utils import dice_score, get_loaders, dice_coeff, multiclass_dice_coeff, dice_loss
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
in_channel = 1
out_channel = 3
learning_rate = 0.0001
model_path = "/media/khanhpham/새 볼륨/unet_kits19_data/checkpoint.pth"
if torch.cuda.is_available():
    device = 'cuda:0'
    print('Running on the GPU')
else:
    device = "cpu"
    print('Running on the CPU')

def test(data, model, device): 
    loop = tqdm(data)
    model.eval()
    tkdice_val = []   
    tudice_val = [] 
    i = 0 
    with torch.no_grad():
        for batch_idx, (image,mask)in enumerate(loop):
            img = image.to(device=device)
            
            mask = mask.to(device=device)
            #writer.add_images("mask_img/0", mask, batch_idx)
            mask = torch.squeeze(mask,dim=1)
            n_mask = F.one_hot(mask, 3).permute(0, 3, 1, 2).float()
            pred = model(img) 
            predictions = F.softmax(pred, dim=1)
            pred_labels = torch.argmax(predictions, dim=1) 
            #pred_labels = torch.unsqueeze(pred_labels,dim=1)
            if n_mask.sum() != 0:
                writer.add_images("img", img, batch_idx)
                writer.add_images("mask_img/gt", n_mask, batch_idx)
                writer.add_images("mask_img/pred", predictions, batch_idx)
            #writer.add_images("mask_img/3", pred_labels, batch_idx)
            tk_dice, tu_dice = dice_score(mask, pred_labels)
            tkdice_val.append(tk_dice)
            tudice_val.append(tu_dice)
            loop.update(img.shape[0])
            loop.set_postfix({"idx": batch_idx})
            loop.set_description("tk_dice:%f|tu_dice:%f"%(tk_dice,tu_dice))
            '''
            if i > 600:
                break
            i +=1
            '''
        return sum(tkdice_val)/len(tkdice_val),sum(tudice_val)/len(tudice_val)

def main():
    loss_vals = []
    test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ])
    test_loader = get_loaders(
        get_dir = "/media/khanhpham/새 볼륨/unet_kits19_data/test/",
        batch_size = 1,
        img_transform = test_transform,
        data_shuffle=True,
    )
    
    model = UNET(in_c=in_channel,out_c=out_channel)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optim"])
    print("Model successfully loaded")

    tk_dice, tu_dice = test(test_loader, model, device)
    print("tk_dice:",tk_dice)
    print("tu_dice:",tu_dice)
    writer.flush()


if __name__ == "__main__":
    main()
