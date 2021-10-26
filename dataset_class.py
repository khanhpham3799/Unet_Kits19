import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
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
        sample = {"image":img[0],"mask":img[1]}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
def test():
    test = kits19_dataset(img_dir = "/media/khanhpham/새 볼륨/unet_kits19_data/train/")
    for i in range(len(test)):
        sample = test[i]
        print(sample["image"].shape, sample["mask"].shape)

if __name__ == "__main__":
    test()
