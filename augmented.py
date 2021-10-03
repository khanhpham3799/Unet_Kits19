from pathlib import Path
import nibabel as nib
import os
import cv2
import torch

data = '/kits19/data/'
output = '/kits19/output/'
output_path = Path(output)
if not output_path.exists():
    output_path.mkdir()
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
    if img.dtype == 'uint16':
        img = img.astype('float64')
    img = torch.Tensor(img)
    img = torch.unsqueeze(img,dim=0)
    return img

for i in list_case:
    if i == "case_00210":
        break
    link_data = Path(data + i)
    img_link = Path(output + "imaging")
    seg_link = Path(output + "segmentation")
    if not img_link.exists():
        img_link.mkdir()
    if not seg_link.exists():
        seg_link.mkdir()
    ipath = str(img_link)+ "/" + i + ".pth"
    spath = str(seg_link)+ "/" + i + ".pth"
    if Path(ipath).exists():
        continue
    vol_nii = nib.load(str(link_data / 'imaging.nii.gz'))
    vol_nii = vol_nii.get_data()
    # vol_nii = np.round(vol_nii).astype(np.uint8)
    seg_nii = nib.load(str(link_data / 'segmentation.nii.gz'))
    seg_nii = seg_nii.get_data()
    # seg_nii = np.round(seg_nii).astype(np.uint8)
    img_data = torch.zeros(1,204,204)
    seg_data = torch.zeros(1,204,204)
    for j in range(vol_nii.shape[0]):
        new_img = crop_resize_img(vol_nii[j])
        new_seg = crop_resize_img(seg_nii[j])
        new_img = change_type(new_img)
        new_seg = change_type(new_seg)
        img_data = torch.cat((img_data,new_img), dim=0)
        seg_data = torch.cat((seg_data,new_seg), dim=0)
    img_data = img_data[1:,:,:]
    seg_data = seg_data[1:,:,:]
    torch.save(img_data,str(ipath))
    torch.save(seg_data,str(spath))
    print(ipath)

#load data and divide training, testing set
def load_dataset(data_link, display = False):
  list_case = sorted(os.listdir(data_link))
  data = torch.load(data_link+list_case[0])
  for i in list_case[1:]:
    new = torch.load(data_link + i)
    data = torch.cat((data,new), dim=0)
    if (display):
        print(data.shape)
  return data

img_link = output + "imaging/"
seg_link = output + "segmentation/"
img_dataset = load_dataset(img_link, display=True)
seg_dataset = load_dataset(seg_link, display=False)


def save_checkpoint(dataset, filename):
  print("=> Saving checkpoint")
  torch.save(dataset, filename)
img_savelink = output + "img_dataset.pth"
seg_savelink = output + "seg_dataset.pth"
save_checkpoint(img_dataset,img_savelink)
save_checkpoint(seg_dataset,seg_savelink)
print("imaging set:",img_savelink)
print("segmentation set:",img_savelink)
