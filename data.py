
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

DATA_DIR = os.getcwd() + "/orange/org/"

def im_to_mask(arr):
    uniques = set()
    new_arr = np.zeros((len(arr[0]), len(arr[1])))
    for i in range(len(arr[0])):
        for j in range(len(arr[1])):
            match t:
                case (34, 167, 132, 255):
                    new_arr[i][j] = 1
                case(68, 1, 84, 255):
                    new_arr[i][j] = 2
                case (253, 231, 36, 255):
                    new_arr[i][j] = 3
                case (64, 67, 135, 255):
                    new_arr[i][j] = 4
                case _:
                    new_arr[i][j] = 0
            # arr[i][j] = 

    mask = torch.tensor(new_arr, dtype=float)
    return mask
class MRIDATASET(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):
        assert mode in {"train", "test"}
        self.root = root
        self.mode = mode
        self.transform = transform
        self.images_directory = DATA_DIR + "/im/"
        self.masks_directory = DATA_DIR + "/lab/"
        self.filenames = []
        x_train_files, x_test_files, _, _ = train_test_split(os.listdir(self.images_directory), os.listdir(self.masks_directory), train_size=0.9, random_state=4)
        if self.mode == "train":
            self.filenames = x_train_files
        if self.mode == "test":
            self.filenames =  x_test_files
        self.to_tensor = torchvision.transforms.ToTensor()
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # * should return a tensor in CHW orientation
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename)
        mask_path = os.path.join(self.masks_directory, filename)

        # * currently in HWC order pre transpose
        image = torch.tensor(np.array(Image.open(image_path).convert("RGB")).transpose(2,0,1)) # TODO should probably be altered to only have one conversion 

        mask_arr = np.array(Image.open(mask_path))
        mask = torch.tensor((np.array(Image.open(mask_path))/255).astype(np.float32).transpose(2,0,1))

        # TODO Determine appropriate transforms for augmentation
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        functional_mask = mask[0,:,:].unsqueeze(0)
        sample = dict(image=image, mask=functional_mask, display_mask=mask)
        return sample
    def display(self,sample):
        image = sample["image"].numpy().transpose((1,2,0))
        mask = sample["display_mask"].numpy().transpose((1,2,0))

        print(np.shape(image))
        plt.subplot(1,2,1)
        plt.imshow(image) 
        plt.subplot(1,2,2)
        plt.imshow(mask)  
        plt.show()







class MRIDATASET_VIDEOS(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):
        assert mode in {"train", "test"}
        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "labels")
        self.filenames = []
        x_train_files, x_test_files, _, _ = train_test_split(os.listdir(os.path.join(self.root, "images")), os.listdir(os.path.join(self.root, "labels")), train_size=0.9, random_state=4)
        if self.mode == "train":
            self.filenames = x_train_files
        if self.mode == "test":
            self.filenames =  x_test_files
        self.to_tensor = torchvision.transforms.ToTensor()
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # * should return a tensor in CHW orientation
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename)
        mask_path = os.path.join(self.masks_directory, filename)

        # * currently in HWC order pre transpose
        image = torch.tensor(np.array(Image.open(image_path).convert("RGB")).transpose(2,0,1)) # TODO should probably be altered to only have one conversion 
        mask = torch.tensor((np.array(Image.open(mask_path))/255).astype(np.float32).transpose(2,0,1))

        # TODO Determine appropriate transforms for augmentation
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        functional_mask = mask[0,:,:].unsqueeze(0)
        sample = dict(image=image, mask=functional_mask, display_mask=mask)
        return sample
    def display(self,sample):
        image = sample["image"].numpy().transpose((1,2,0))
        mask = sample["display_mask"].numpy().transpose((1,2,0))

        print(np.shape(image))
        plt.subplot(1,2,1)
        plt.imshow(image) 
        plt.subplot(1,2,2)
        plt.imshow(mask)  
        plt.show()



