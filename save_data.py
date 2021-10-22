import nibabel as nib
import os
import numpy as np
data_link = "/media/khanhpham/새 볼륨/kits19/data/"
train_link = "/home/khanhpham/Unet_Kits19/output/train/"
test_link = "/home/khanhpham/Unet_Kits19/output/test/"
list_case = sorted(os.listdir(data_link))
split_ratio = 0.2
patient = 0
for case in list_case:
    img = nib.load(str(case/'imaging.nii.gz'))
    img = img.get_data()
    seg = nib.load(str(case/'segmentation.nii.gz'))
    seg = seg.get_data()
    data_len = img.shape[0]
    img = np.reshape(img, (data_len,1,512,512))
    seg = np.reshape(seg, (data_len,1,512,512))
    print(img.shape)
    for i in range(0,data_len):
        np_data = np.stack((img[i],seg[i]),axis = 0)
        print(np_data.shape)
        if random.random() < 0.2:
            np.save(test_link+str(patient)+"_"+str(i),np_data)
        else:
            np.save(train_link+str(patient)+"_"+str(i),np_data)
    patient+=1
