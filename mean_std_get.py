import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
img_path = "/media/khanhpham/새 볼륨/unet_kits19_data/train/"
model_path = "/media/khanhpham/새 볼륨/unet_kits19_data/mean_std_get.pth"
list_img = sorted(os.listdir(img_path))
org_img = np.load(str(os.path.join(img_path,list_img[0])))
org_img = org_img[0]
i = 0
for img in list_img[1:]:
    img_link = os.path.join(img_path,img)
    temp_img = np.load(str(img_link))
    temp_img = temp_img[0]
    org_img = np.vstack((org_img,temp_img))
    print(org_img.shape)
    if i > 10000:
        break
    i+=1
mean = np.mean(org_img)
std = np.std(org_img)
print("mean:",mean,"std:",std)
torch.save({
    "mean": mean,
    "std": std,
    }, model_path
    )
print("Finish saving!")