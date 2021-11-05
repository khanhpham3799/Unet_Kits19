import torch
import torchvision
from dataset_class import kits19_dataset
from torch.utils.data import DataLoader
from torch import Tensor

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
def dice_score(gt, pred, ep = 1e-6):
    # Compute tumor+kidney Dice
    tk_pd = torch.greater(pred,0)
    tk_gt = torch.greater(gt,0)
    tk = tk_pd.sum()+tk_gt.sum()
    tk_dice = (2*(torch.logical_and(tk_pd,tk_gt).sum()) + ep) / (tk+ep)
    # Compute tumor Dice
    tu_pd = torch.greater(pred, 1)
    tu_gt = torch.greater(gt, 1)
    tu = tu_pd.sum() + tu_gt.sum()
    tu_dice = (2*torch.logical_and(tu_pd, tu_gt).sum() +ep)/ (tu+ep)
    return tk_dice, tu_dice

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)    