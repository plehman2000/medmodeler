#RUNNING ON ENVIRONMENT 'UNEXT'
# Config
seed = 42  # for reproducibility
training_split_ratio = 0.9  # use 90% of samples for training, 10% for testing
num_epochs = 5
# If the following values are False, the models will be downloaded and not computed
compute_histograms = False
train_whole_images = False 
train_patches = False
import uuid
import enum
import time
import random
import multiprocessing
from pathlib import Path
from tqdm import tqdm
import torch
torch.cuda.empty_cache()

import torchvision
import torchio as tio
import torch.nn.functional as F

import numpy as np
from unet import UNet
from scipy import stats
import matplotlib.pyplot as plt

from IPython.display import display
from tqdm.auto import tqdm

random.seed(seed)
torch.manual_seed(seed)
plt.rcParams['figure.figsize'] = 12, 6

import pytorch_lightning as pl
import os
import time




import torch
import torchio
import pytorch_lightning as pl
import wandb
from pathlib import Path
from tqdm import tqdm

PATH = "/home/patricklehman/MRI"
import numpy as np
def numpy_reader(path):
    data = np.load(path, allow_pickle=True)
    affine = np.eye(4)
    return data, affine

class MedicalImageDataset(torch.utils.data.Dataset):
    def __init__(self, subjects_dataset):
        self.dataset = subjects_dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        x = self.dataset[idx]
        inputs = x['sample']
        targets = x['label']
        print(inputs.shape, targets.shape)
        return inputs, targets

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

def collate_fn(batch):
#     batch = list(filter(lambda x: x is not None, batch))
#     print(batch)
    return torch.utils.data.dataloader.default_collate(batch)



class CATDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self._has_setup_0 = True
        self.image_shape = wandb.config['image_shape']
        self.training_subjects = []
        self.validation_subjects = []
        self.batch_size = batch_size
        self.dataset = []
        self.num_workers = wandb.config['num_workers']
        self.training_split_ratio = 0.9
        self.persistent_workers = wandb.config['persistent_workers']
        self.debug = True
        
    def prepare_data(self):
        print("prepping data")
        dataset_dir = Path(PATH +  "/orange/org/augfix") #switched from augfix due to one hot error
        images_dir = dataset_dir / 'images'
        labels_dir = dataset_dir / 'labels'
        image_paths = sorted(images_dir.glob('*.npy'))
        label_paths = sorted(labels_dir.glob('*.npy'))
        assert len(image_paths) == len(label_paths)
        augmented_dataset = []
        count = 0
        for (image_path, label_path) in tqdm(zip(image_paths, label_paths), total=len(image_paths)):
            subject = tio.Subject(
                sample=tio.ScalarImage(image_path, reader = numpy_reader),
                label =  tio.LabelMap(label_path, reader = numpy_reader),transform = tio.CropOrPad(self.image_shape))
            
            self.dataset.append({"sample":subject['sample'][tio.DATA].contiguous(), "label":subject['label'][tio.DATA].contiguous()})
            del subject#TODO
            count +=1
#             print(count)
            if self.debug and count ==2: #debugging DO NOT LEAVE TODO
                break

    def setup(self, stage):
        print("setting up")
        num_training_subjects = int(self.training_split_ratio * len(self.dataset))
        num_validation_subjects = len(self.dataset) - num_training_subjects
        num_split_subjects = [num_training_subjects, num_validation_subjects]
        #WHY IS THIS NUMBER WRONG?
        self.training_subjects, self.validation_subjects = torch.utils.data.random_split(self.dataset, num_split_subjects)
#       



    def train_dataloader(self):
#         training_set = tio.SubjectsDataset(self.training_subjects, transform = tio.CropOrPad(self.image_shape))
        training_set = MedicalImageDataset(self.training_subjects)
        training_loader = torch.utils.data.DataLoader(
                            training_set,
                            batch_size=self.batch_size ,
                            shuffle=True,
                            num_workers=self.num_workers,
                            worker_init_fn=seed_worker,
                            persistent_workers=self.persistent_workers,
                            collate_fn=collate_fn)
            
        return training_loader

    def val_dataloader(self):
        validation_set = MedicalImageDataset(self.validation_subjects)
        
#         validation_set = tio.SubjectsDataset(self.validation_subjects,transform = tio.CropOrPad(self.image_shape))
        validation_loader = torch.utils.data.DataLoader(
                        validation_set,
                        batch_size=self.batch_size ,
                        shuffle=False,
                        num_workers=self.num_workers,
                        worker_init_fn=seed_worker,
                        persistent_workers=self.persistent_workers,
                        collate_fn=collate_fn)
        
        return validation_loader

print('runs')