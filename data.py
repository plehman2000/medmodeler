
#figure out format of annotation images
from sklearn.model_selection import train_test_split
import torch
import os
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pathlib

import torchvision
import random
random.seed(10)
import logging
logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.INFO)
logger = logging.getLogger()
logger.disabled = True

DATA_DIR = os.getcwd() + "/orange/org/"
N_CLASSES = 5
def im_to_mask(arr):
    new_arr = np.zeros((len(arr[0]), len(arr[1]),N_CLASSES)) #last dim is num classes - 1
    for i in range(len(arr[0])):
        for j in range(len(arr[1])):
                # t = tuple([int(x) for x in arr[i][j]])
                t = tuple(arr[i][j])
                if t== (34, 167, 132, 255):
                    new_arr[i][j] = [1.,0.,0.,0.,0.]
                elif t== (68, 1, 84, 255):
                    new_arr[i][j] = [0.,1.,0.,0.,0.]
                elif t==  (253, 231, 36, 255):
                    new_arr[i][j] = [0.,0.,1.,0.,0.]
                elif t==  (64, 67, 135, 255):
                    new_arr[i][j] = [0.,0.,0.,1.,0.]
                elif t == (41, 120, 142, 255):
                    new_arr[i][j] = [0.,0.,1.,0.,0.]
                else:
                    # print(f"{t} is {type(t)}| {t[0]} is {type(t[0])}")
                    new_arr[i][j] = [0.,0.,0.,0.,0.] #ENSURE THE CHANGE TO ALL ZEROS IS PROPAGATED
            # arr[i][j] = 
    new_arr = new_arr.transpose(2,0,1)
    del arr
    # print(f"Uniques: {uniques}")
    
    return new_arr

def image_filter(directory_list):
    return [x for x in directory_list if x.endswith(".png") ]


class MRIDATASET(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None, clip_to=None, shuffle=True):
        assert mode in {"train", "test"}
        self.root = root
        self.mode = mode
        self.transform = transform
        self.images_directory = DATA_DIR + "/im/"
        self.masks_directory = DATA_DIR + "/lab/"
        self.filenames = []
        # print(len(os.listdir(self.images_directory)) ,len(os.listdir(self.masks_directory)))
        assert len(os.listdir(self.images_directory)) == len(os.listdir(self.masks_directory))
        x_train_files, x_test_files, _, _ = train_test_split(image_filter(os.listdir(self.images_directory)), image_filter(os.listdir(self.masks_directory)), train_size=0.9, random_state=4)
        if self.mode == "train":
            self.filenames = x_train_files
        if self.mode == "test":
            self.filenames =  x_test_files
        self.to_tensor = torchvision.transforms.ToTensor()
        if clip_to is not None:
            if shuffle:
                self.filenames = random.sample(self.filenames, clip_to)
            else:
                self.filenames = self.filenames[:clip_to]
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # * should return a tensor in CHW orientation
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename)
        mask_path = os.path.join(self.masks_directory, filename)

        # * currently in HWC order pre transpose
        sample_image = Image.open(image_path)
        sample_mask = Image.open(mask_path)
        if self.transform is not None:
            sample_image = self.transform(sample_image)
            sample_mask = self.transform(sample_mask)
        
        image = torch.tensor(np.array(sample_image.convert("RGB")).transpose(2,0,1)) # TODO should probably be altered to only have one conversion 
        mask_arr = np.array(sample_mask)
        logging.info("Pre im_to_mask shape: %s",{mask_arr.shape})
                
        new_arr = im_to_mask(mask_arr)
        mask = torch.tensor(new_arr, dtype=float)
        # mask = mask.unsqueeze(0)
        logging.info("Post im_to_mask shape: %s",mask.shape)
        
        # TODO Determine appropriate transforms for augmentation
        
        logging.info("Post transform shape: %s",mask.shape)
        sample = dict(image=image, mask=mask, display_mask=mask)
        return sample







class MRIDATASET_PILLOW(torch.utils.data.Dataset):
    def __init__(self, images:list, transform):
        self.images= images
        self.transform = transform
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # * currently in HWC order pre transpose
        image = torch.tensor(np.array(self.images[idx].convert("RGB")).transpose(2,0,1)) # TODO should probably be altered to only have one conversion 
        if self.transform is not None:
            image = self.transform(image)
        return {"image":image}




# def mask_to_im(arr):
#     new_arr = np.zeros((len(arr[0]), len(arr[1]),3))
#     lab  = 0
#     for i in range(len(arr[0])):
#         for j in range(len(arr[1])):
#             label = np.argmax(arr[i][j])
#             if label == 0:
#                 new_arr[i][j] = [34, 167, 132]
#             elif label == 1:
#                 new_arr[i][j] = [68, 1, 84]
#             elif label == 2:
#                 new_arr[i][j] = [76, 231, 255]
#             elif label == 3:
#                 new_arr[i][j] = [64, 67, 135]
#             elif label ==4:
#                 new_arr[i][j] == [41, 120, 142]
#             else:
#                 new_arr[i][j] = [0, 0, 0]
#     del arr
#     new_arr = new_arr.astype(np.uint8)
#     return new_arr