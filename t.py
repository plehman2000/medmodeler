import torch
from PIL import Image
import numpy as np
import os

mask_path = os.getcwd() + "\orange\org\lab\S0S_60.png" 

mask = torch.tensor((np.array(Image.open(mask_path))/255).astype(np.float32).transpose(2,0,1))
print(mask)
print(mask.shape)
# ! fix to allow for multi class work
