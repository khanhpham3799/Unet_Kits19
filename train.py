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
learning_rate = 0.000001
load_model = True
test_run = False
model_path = "/media/khanhpham/새 볼륨/unet_kits19_data/checkpoint.pth"
if torch.cuda.is_available():
    device = 'cuda:1'
    print('Running on the GPU')
else:
    device = "cpu"
    print('Running on the CPU')

def train(data, model, optimizer, loss_fn, scaler, device):
    loop= tqdm(data)
    model.train()
    dice = []
    loss_list = []
    for batch_idx, (img, mask) in enumerate(loop):
        img = img.to(device=device, dtype = torch.float)
        mask = mask.to(device=device)
        if mask.sum() == 0:
            if random.random() < 0.5:
                print("skip")
                continue
        mask = torch.squeeze(mask, dim=1)
        n_mask = F.one_hot(mask, 3).permute(0, 3, 1, 2).float()
        with torch.cuda.amp.autocast():
            pred = model(img)
            pred = F.softmax(pred, dim=1).float()
            pred_labels = torch.argmax(pred, dim=1) 
            loss = dice_loss(pred,n_mask, multiclass=True)
            loss_list.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer) 
        scaler.update()
        tk_dice, tu_dice = dice_score(mask, pred_labels)
        dice.append(tk_dice)
        loop.update(img.shape[0])
        loop.set_postfix({"idx":batch_idx})
        loop.set_description("Loss:%.5f|Dice: %.5f"%(loss.item(),tk_dice))
    return sum(loss_list)/len(loss_list), sum(dice)/(len(dice))

def test(data, model, device): 
    loop = tqdm(data)
    model.eval()
    tkdice_val = []   
    tudice_val = []
    with torch.no_grad():
        for batch_idx, (image,mask)in enumerate(loop):
            img = image.to(device=device)
            mask = mask.to(device=device)
            #(64,1,512,512)->(64,512,512)
            mask = torch.squeeze(mask,dim=1) 
            predictions = model(img) 
            predictions = F.softmax(predictions, dim=1)
            #(64,3,512,512)->(64,512,512)
            pred_labels = torch.argmax(predictions, dim=1) 
            tk_dice, tu_dice = dice_score(mask, pred_labels)
            tkdice_val.append(tk_dice)
            tudice_val.append(tu_dice)
            loop.update(img.shape[0])
            loop.set_postfix({"idx": batch_idx})
            loop.set_description("tk_dice:%f|tu_dice:%f"%(tk_dice,tu_dice))
        return sum(tkdice_val)/len(tkdice_val), sum(tudice_val)/len(tudice_val)

def main():
    loss_vals = []
    train_transform = transforms.Compose(
    [
        #transforms.ToPILImage(),
        #transforms.Resize((128,128)),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose(
    [
        #transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    train_loader = get_loaders(
        get_dir = "/media/khanhpham/새 볼륨/unet_kits19_data/train/",
        batch_size = 8,
        img_transform = train_transform,
        data_shuffle=True,
    )
    test_loader = get_loaders(
        get_dir = "/media/khanhpham/새 볼륨/unet_kits19_data/test/",
        batch_size = 1,
        img_transform = test_transform,
        data_shuffle=False,
    )

    model = UNET(in_c=in_channel,out_c=out_channel)
    model = model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=1)
    if load_model: 
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optim"])
        epoch = checkpoint["epoch"]+1
        loss_vals = checkpoint["loss_values"]
        dice_list = checkpoint["dice_list"]
        dice = checkpoint["dice_score"]
        print("Model successfully loaded")
    else:
        epoch = 0
        loss_vals = []
        dice_list = []
        dice = 0
        tk_dice = 0
    for i in range(epoch, 10):
        print(f"Epoch: {i}")
        print("dice_score:", dice)    
        loss_val, train_dice = train(train_loader, model,optimizer,loss_fn, scaler, device)
        writer.add_scalar("tk_dice", train_dice, i)
        writer.add_scalar("loss", loss_val, i)
        loss_vals.append(loss_val)
        scheduler.step(train_dice)

        if test_run:
            tk_dice, tu_dice = test(test_loader, model, device)
            scheduler.step(tk_dice)
            writer.add_scalar("test_dice", tk_dice, i)
            dice_list.append(tk_dice)
            print("average test dice score:", tk_dice)
            print("average test tu_dice score:", tu_dice)
        else:
            tk_dice = dice
        if tk_dice >= dice:
            torch.save({
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "epoch": i,
                "loss_values": loss_vals,
                "dice_list": dice_list,
                "dice_score": tk_dice
            }, model_path
            )
            dice = tk_dice
        print("average train dice score:", train_dice)       
        print("Finish training and saved model!")
        writer.flush()

if __name__ == "__main__":
    main()
