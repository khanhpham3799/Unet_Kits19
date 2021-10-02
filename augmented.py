from pathlib import Path
import nibabel as nib
import os
import cv2
import torch

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
    if img.dtype == 'uint16':
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

#load data and divide training, testing set
def load_dataset(data_link, data_type, display = False):
  list_case = sorted(os.listdir(data_link))
  data_path = Path(data_link)
  initial_data = torch.zeros(1,204,204)
  for i in list_case:
    link = str(data_link + i + '/' + data_type)
    data_load = torch.load(link)
    initial_data = torch.cat((initial_data,data_load), dim=0)
    if (display):
      print(i,initial_data.shape)
  initial_data = initial_data[1:,:,:]
  return initial_data

data = '/kits19/output/'
img_dataset = load_dataset(data,data_type = 'imaging', display=True)
seg_dataset = load_dataset(data,data_type = 'segmentation', display=False)


def save_checkpoint(dataset, filename):
  print("=> Saving checkpoint")
  torch.save(dataset, filename)
img_savelink = "/kits19/img_dataset.pth"
seg_savelink = "/kits19/seg_dataset.pth"
save_checkpoint(img_dataset,img_savelink)
save_checkpoint(seg_dataset,seg_savelink)
print("imaging set:",img_savelink)
print("segmentation set:",img_savelink)