from Unet_model import build_unet
from evaluate import dice_loss, dice_coeff, multiclass_dice_coeff
import random
import torch
import torchvision
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import DataLoader,Dataset, TensorDataset # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.utils as utils
import wandb
from torch import Tensor
from torchsummary import summary

learning_rate = 0.001
batch_size = 10
val_percent = 0.3
global_step = 0

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load data from tensor file and creating dataset
img_savelink = "/kits19/img_dataset.pth"
seg_savelink = "/kits19/seg_dataset.pth"
img = torch.load(img_savelink)
seg = torch.load(seg_savelink)
#divide training and test set, transform data to tensordataset
X_train, X_test, y_train, y_test = train_test_split(img, seg, test_size=val_percent,random_state=42)
train_set = TensorDataset(X_train,y_train)
train_loader = DataLoader(dataset=train_set, batch_size = batch_size, shuffle = True)
test_set = TensorDataset(X_test,y_test)
test_loader = DataLoader(dataset=test_set, batch_size = batch_size, shuffle = True)

input_channel = 1
output_channel = 3  # 3 classes 0,1,2
net = build_unet(in_c=input_channel, out_c=output_channel)
net.to(device)
optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
# reduce the learning_rate by 10 times (factor = 0.1) when in 2 epoch, the value of acc have not increased (max)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=10)  # goal: maximize Dice score
# prevent gradient value to be zero
grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
criterion = nn.CrossEntropyLoss()


def train(epoch, train_loader):
    print('\nEpoch: %d' % epoch)
    net.train()
    global_step = 0
    epoch_loss = 0
    dice_score = 0
    img_no = 0
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for batch_idx, (images, true_masks) in enumerate(train_loader):
            images = torch.unsqueeze(images, dim=1)  # (10,1,204,204)
            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            # check if the batch_size have all 0 masks
            if true_masks.sum() == 0:
                # generate a random number from 0 to 1
                if random.random() < 0.9:
                    # 90% skip the iteration
                    continue
            # use mix precision data type
            with torch.cuda.amp.autocast(enabled=True):
                masks_pred = net(images)  # (10,3,204,204)
                # rescale the element to range [0,1] along dimension1
                input = F.softmax(masks_pred, dim=1).float()  # 10,3,204,204
                # (10,204,204) -> F.one_hot (10,204,204,3) -> permute (10,3,204,204)
                # f.onehot() return 1 at the index that has class value (0,2,2,0)->(0,1,1,0)
                target = F.one_hot(true_masks, 3).permute(0, 3, 1, 2).float()
                loss = criterion(masks_pred, true_masks) \
                       + dice_loss(input, target, multiclass=True)

                pred_target = F.one_hot(masks_pred.argmax(dim=1), 3).permute(0, 3, 1, 2).float()
                dice = multiclass_dice_coeff(pred_target[:, 1:, ...], target[:, 1:, ...], reduce_batch_first=False)
                dice_score += dice
                img_no += images.shape[0]

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()  # multiple with scale factor(loss) in backward
            grad_scaler.step(optimizer)  # optimizer.step()
            grad_scaler.update()  # update the scale factor

            for param_group in optimizer.param_groups:
                new_lr = param_group['lr']

            pbar.update(images.shape[0])
            global_step += 1
            epoch_loss += loss.item()
            pbar.set_postfix({'batch_idx': batch_idx})
            pbar.set_description("Epoch_loss: %.3f | Dice_score: %.3f" % (loss.item(), dice_score / img_no))

test_dice = 0
def test(epoch, test_loader):
    global test_dice
    net.eval()
    dice_score = 0
    img_no = 0
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
            for batch_idx, (images, true_masks) in enumerate(test_loader):
                images = torch.unsqueeze(images, dim=1)
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                mask_true = F.one_hot(true_masks, 3).permute(0, 3, 1, 2).float()  # 10,3,204,204
                mask_pred = net(images)  # 10,3,204,204
                # mask_pred.argmax(dim=1) return the channel that have the greatest index's value in each index
                # 10,3,204,204 -> 10,204,204 -> one_hot 10,204,204,3 -> permute 10,3,204,204
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), 3).permute(0, 3, 1, 2).float()
                # calculate from channel 1 (eliminate channel full 0)
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)
                img_no += images.shape[0]
                pbar.update(images.shape[0])
                pbar.set_postfix({'batch_idx': batch_idx})
                pbar.set_description("Test_dicescore: %.3f " % (dice_score / img_no))
    dice = dice_score / len(test_loader)
    if dice >= test_dice:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, 'Unet_checkpoint.pth')
        torch.save(dice, 'Unet_bestdicescore.pth')
        test_dice = dice
    return dice


start_epoch = 0
num_epoch = 1
for epoch in range(start_epoch, start_epoch + num_epoch):
    train(epoch, train_loader)
    test_dice = test(epoch, test_loader)
    scheduler.step(test_dice)
