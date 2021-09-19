# !git clone https://github.com/neheller/kits19
# %cd /kits19
import starter_code
from starter_code.utils import load_case
from starter_code.visualize import visualize, hu_to_grayscale
from pathlib import Path
import nibabel as nib
import numpy as np
import scipy
import skimage
from scipy import ndimage
from skimage.transform import resize
import imageio
from imageio import imwrite
import os
import cv2

data = '/kits19/data/'
output = '/kits19/output/'
new_data = Path(data)
cases = sorted([d for d in new_data.iterdir() if d.is_dir()])
list_case = sorted(os.listdir(data))

def crop_resize_img(img):
    row = img.shape[0]
    col = img.shape[1]
    new_img = img[int(row / 10):int(row * 9 / 10), int(col / 10):int(col * 9 / 10)]
    new_image = cv2.resize(new_img, (int(new_img.shape[0] / 2), int(new_img.shape[1] / 2)))
    return new_image

def change_type(img):
    if img.dtype == 'unit16':
        img = img.astype('float64')
    return img

for i in list_case:
    output_case = Path(output + i)
    link_data = Path(data + i)
    if output_case.exists():
        continue
    if not output_case.exists():
        output_case.mkdir()
    vol_nii = nib.load(str(link_data / 'imaging.nii.gz'))
    vol_nii = vol_nii.get_data()
    # vol_nii = np.round(vol_nii).astype(np.uint8)
    seg_nii = nib.load(str(link_data / 'segmentation.nii.gz'))
    seg_nii = seg_nii.get_data()
    # seg_nii = np.round(seg_nii).astype(np.uint8)
    img_link = Path(output + i + "/imaging")
    seg_link = Path(output + i + "/segmentation")
    if not img_link.exists():
        img_link.mkdir()
    if not seg_link.exists():
        seg_link.mkdir()
    for j in range(vol_nii.shape[0]):
        ipath = Path(str(img_link) + "/{:05d}.pth".format(j))
        spath = Path(str(seg_link) + "/{:05d}.pth".format(j))
        new_img = crop_resize_img(vol_nii[j])
        new_seg = crop_resize_img(seg_nii[j])
        new_img = change_type(new_img)
        new_seg = change_type(new_seg)
        torch.save(new_img,str(ipath))
        torch.save(new_seg,str(spath))
    print(output_case)