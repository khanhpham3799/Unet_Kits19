import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from Unet_model import UNET
from utils import dice_score, get_loaders
in_channel = 1
out_channel = 3
learning_rate = 0.0001
load_model = False
epoch = 0
epochs = 2
dice = 0
model_path = "/media/khanhpham/새 볼륨/unet_kits19_data/checkpoint.pth"
if torch.cuda.is_available():
    device = 'cuda:1'
    print('Running on the GPU')
else:
    device = "cpu"
    print('Running on the CPU')

def train(data,model,optimizer,loss_fn,device):
    loop= tqdm(data)
    model.train()
    for batch_idx, (img, mask) in enumerate(loop):
        img = img.to(device=device, dtype = torch.float)
        mask = mask.to(device=device)
        mask = torch.squeeze(mask, dim=1)
        mask = F.one_hot(mask, 3).permute(0, 3, 1, 2).float()
        if mask.sum() == 0:
            if random.random() < 0.8:
                print("skip")
                continue
        with torch.cuda.amp.autocast():
            pred = model(img)
            loss = loss_fn(pred,mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loop.update(img.shape[0])
        loop.set_postfix({"idx":batch_idx})
        loop.set_description("Loss:%3f"%(loss.item()))
    return loss.item()

def test(data, model, device): 
    loop = tqdm(data)
    model.eval()
    dice_val = []   
    with torch.no_grad():
        for batch_idx, (image,mask)in enumerate(loop):
            img = image.to(device=device)
            mask = mask.to(device=device)
            predictions = model(img) 
            predictions = F.softmax(predictions, dim=1)
            #(64,3,512,512)->(64,512,512)
            pred_labels = torch.argmax(predictions, dim=1) 
            pred_labels = pred_labels.float()
            tk_dice, tu_dice = dice_score(mask,pred_labels)
            if tk_dice != 0:
                dice_val.append(tk_dice)
            loop.update(img.shape[0])
            loop.set_postfix({"idx": batch_idx})
            loop.set_description("tk_dice:%3f|tu_dice:%3f"%(tk_dice,tu_dice))
        return sum(dice_val)/len(dice_val)

def main():
    global epoch
    global dice
    loss_vals = []
    train_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((128,128)),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ])
    train_loader = get_loaders(
        get_dir = "/media/khanhpham/새 볼륨/unet_kits19_data/train/",
        batch_size = 32,
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
    
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
    if load_model: 
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optim"])
        epoch = checkpoint["epoch"]+1
        loss_vals = checkpoint["loss_values"]
        dice = checkpoint["dice_score"]
        print("Model successfully loaded")

    for i in range(epoch, epochs):
        print(f"Epoch: {i}")
        loss_vals = []
        loss_val = train(train_loader, model,optimizer,loss_fn, device)
        avg_dice = test(test_loader, model, device)
        scheduler.step(avg_dice)
        loss_vals.append(loss_val)
        if avg_dice > dice:
            torch.save({
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "epoch": i,
                "loss_values": loss_vals,
                "dice_score": avg_dice
            }, model_path
            )
            dice = avg_dice
        print("average dice score:", avg_dice)
        print("Finish training and saved model!")

if __name__ == "__main__":
    main()
