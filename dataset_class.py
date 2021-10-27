import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
class kits19_dataset(Dataset):
    def __init__(self,img_dir,transform=None):
        self.dir = img_dir
        self.transform = transform
        self.img = sorted(os.listdir(img_dir))
    
    def __len__(self):
        return len(self.img)

    def __getitem__(self,index):
        img_path = os.path.join(self.dir,self.img[index])
        img = np.load(str(img_path))
        image = img[0]
        image = image[0].astype(np.float32)
        mask = img[1]
        mask = mask[0].astype(np.float32)
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        mask = mask.type(torch.LongTensor)
        return image, mask
def test():
    test = kits19_dataset(img_dir = "/media/khanhpham/새 볼륨/unet_kits19_data/train/")
    for i in range(len(test)):
        sample = test[i]
        print(i,sample["image"].shape, sample["mask"].shape)

if __name__ == "__main__":
    test()




