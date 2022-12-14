import torch
import numpy as np
import os

samp = np.load(os.getcwd() + "/DATA/sample/sample0.npy")
label = np.load(os.getcwd() + "/DATA/label/label0.npy")
SLICE_START = 0
SLICE_LENGTH = 50
sample = torch.tensor(samp)
label = torch.tensor(label)
sample = torch.narrow(sample, 3, SLICE_START, SLICE_LENGTH)
label = torch.narrow(label, 3, SLICE_START, SLICE_LENGTH)
lab0 = torch.where(label==0,1,0);lab1 = torch.where(label==1,1,0)
lab2 = torch.where(label==2,1,0);lab3 = torch.where(label==3,1,0)
lab4 = torch.where(label==4,1,0);lab5 = torch.where(label==5,1,0)
master_label = torch.stack([lab0,lab1,lab2,lab3,lab4,lab5], dim = 1)
print(master_label.shape, torch.unique(master_label))