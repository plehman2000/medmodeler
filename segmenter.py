
seed = 42  
import uuid
from IPython.display import display
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
# torch.multiprocessing.set_start_method('spawn', force=True)

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
  
import torchvision.transforms.functional as Ft
from utilities import *
from losses import *
from segmenter import *
CHANNELS_DIMENSION = 1
SPATIAL_DIMENSIONS = 2, 3, 4
from unet import UNet2D, UNet3D
import torch.nn as nn
MODEL_PATH = "/home/patricklehman/MRI/model"
import wandb
# import datamodule
from monai.networks.nets import SwinUNETR
class Segmenter(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Choose the `slow_r50` model 
#         model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        self.model = UNet3D(
        in_channels=1,
        out_classes=6,
        out_channels_first_layer=wandb.config['out_channels'],
        normalization=wandb.config['normalization'], 
        preactivation=wandb.config['preactivation'],
        residual=wandb.config['residual'],
        num_encoding_blocks=wandb.config['num_encoding_blocks'],
        upsampling_type='trilinear')
#         self.model=model = SwinUNETR(img_size=wandb.config['image_shape'],
#                   in_channels=1,
#                   out_channels=6,
#                   feature_size=24,
#                   use_checkpoint=False,
#                   )
        self.train_loss = 0
        self.val_loss = 0
        self.criterion = wandb.config['criterion']()#nn.CrossEntropyLoss()
        
        
        
        
    def forward(self, x):
        return self.model(x)
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=wandb.config['learning_rate'])
        return optimizer
    def training_step(self, train_batch, batch_index):
        inputs, targets = train_batch
            
        with torch.enable_grad():
            logits = self.model(inputs)
#             probabilities = F.softmax(logits, dim=CHANNELS_DIMENSION)
#             batch_losses = self.criterion(probabilities, targets)
            batch_losses = self.criterion(logits, targets)
            batch_loss = batch_losses.mean()
        self.log("train/loss", batch_loss, sync_dist=True, batch_size=wandb.config['batch_size'])
        
        return batch_loss
   
    def validation_step(self, val_batch, batch_index):
        inputs, targets = val_batch


        with torch.no_grad():
            logits = self.model(inputs)
#             probabilities = F.softmax(logits, dim=CHANNELS_DIMENSION)
#             batch_losses = self.criterion(probabilities, targets)

            batch_losses = self.criterion(logits, targets)
            batch_loss = batch_losses.mean()
#         dic = {'val_loss': batch_loss}
#         self.log(dic)
        self.log("val/loss", batch_loss,sync_dist=True,  batch_size=wandb.config['batch_size'])
        tag = time.ctime().replace(":", "_").replace(" ", "_")
        torch.save({'state_dict': self.model.state_dict()}, f"{MODEL_PATH}/{batch_index}{str(uuid.uuid4())[:5]}")
        return batch_loss
    
    def predict_step(self, batch=None,raw_tens=None, batch_idx=0, datloader_idx=0):
        if raw_tens != None:
            inputs = raw_tens
        else:
            inputs, targets = batch_to_targ_input(batch)

        with torch.no_grad():
            logits = self.model(inputs)
#             probabilities = F.softmax(logits, dim=CHANNELS_DIMENSION)
            if raw_tens != None:
                return probabilities
            batch_losses = self.criterion(probabilities, targets)
            batch_loss = batch_losses.mean()
        return inputs, targets, probabilities, batch_loss
