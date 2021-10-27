import torch
import torchvision
from dataset_class import kits19_dataset
from torch.utils.data import DataLoader


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    get_dir,
    batch_size,
    img_transform,
    data_shuffle,
    ):
    data = kits19_dataset(
        img_dir = get_dir,
        transform = img_transform,
    )
    data_loader = DataLoader(
        data,
        batch_size = batch_size,
        shuffle = data_shuffle,
    )
    return data_loader
def dice_score(gt, pred):
    try:
        # Compute tumor+kidney Dice
        tk_pd = torch.greater(pred,0)
        tk_gt = torch.greater(gt,0)
        tk_dice = 2*torch.logical_and(tk_pd,tk_gt).sum()/(tk_pd.sum()+tk_gt.sum())
    except ZeroDivisionError:
        return 0.0, 0.0

    try:
        # Compute tumor Dice
        tu_pd = torch.greater(pred, 1)
        tu_gt = torch.greater(gt, 1)
        tu_dice = 2*torch.logical_and(tu_pd, tu_gt).sum()/(
            tu_pd.sum() + tu_gt.sum()
        )
    except ZeroDivisionError:
        return tk_dice, 0.0

    return tk_dice, tu_dice

    