


#   FILTER MRI BY VALUE?

import torchvision.transforms as T
transform = T.ToPILImage()
import PIL
import cv2

import torch
import numpy as np
import os

samp = np.load(os.getcwd() + "/DATA/sample/sample0.npy")
label = np.load(os.getcwd() + "/DATA/label/label0.npy")
SLICE_START = 0
SLICE_LENGTH = 64
sample = torch.tensor(samp)

label = torch.tensor(label)
sample = torch.narrow(sample, 3, SLICE_START, SLICE_LENGTH)
label = torch.narrow(label, 3, SLICE_START, SLICE_LENGTH)
lab0 = torch.where(label==0,1,0);lab1 = torch.where(label==1,1,0)
lab2 = torch.where(label==2,1,0);lab3 = torch.where(label==3,1,0)
lab4 = torch.where(label==4,1,0);lab5 = torch.where(label==5,1,0)
master_label = torch.stack([lab0,lab1,lab2,lab3,lab4,lab5], dim = 1)

sample = torch.reshape(sample, ( 1,1,sample.shape[3], sample.shape[1], sample.shape[2]))
master_label = torch.reshape(master_label, (1, master_label.shape[1],master_label.shape[4], master_label.shape[2], master_label.shape[3]))
print(master_label.shape[1])

print( sample.shape, master_label.shape)

from matplotlib import cm

#Visualizing tensor functions
def write_video(file_path, frames, fps, grayscale=False):
    """
    Writes frames to an mp4 video file
    :param file_path: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    """
    w, h = frames[0].size
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))

    for frame in frames:
        open_cv_image = np.asarray(frame)
        if not grayscale:
            # Convert RGB to BGR 
            open_cv_image = open_cv_image[:, :, ::-1]
           
        else:
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_GRAY2BGR)
        
        writer.write(open_cv_image)

    writer.release() 

def sample_to_video(inputs, name="label_vid", repeat=3):
    inputs = inputs[0]
    slices = []
    for i in range(inputs.size()[2]):
        new = inputs[:,:,i]
        pil_image = transform(new)
        for _ in range(repeat):
            slices.append(pil_image)
        
    return slices


def label_to_video(inputs, name="label_vid", repeat=3):
    """
    Given an output tensor from the network with shape [6, 256, 256, 10]), return a
    video of the tensor with name "name"
    """
    num_classes=6
    color = (torch.ones((1,inputs.shape[1],inputs.shape[2],inputs.shape[3]))/num_classes)
    re_colored = inputs * color
    re_colored = re_colored[0]
    slices = []
    magma = cm.get_cmap('YlGnBu')
    for i in range(re_colored.size()[2]):
        new = re_colored[:,:,i]
        new = magma(new)
        new = new[:,:,:3]*255
        pil_image = PIL.Image.fromarray(new.astype(np.uint8))
#         display(pil_image)
        for _ in range(repeat):
            slices.append(pil_image)
    return slices



#sample
img_size = (SLICE_LENGTH, 256, 256)


sample = torch.ones(1, 1, img_size[0], img_size[1], img_size[2])
print(sample.size())
slices=sample_to_video(sample)
write_video(os.getcwd() + "/m.mp4", slices, 10, True)

import pytorch_lightning as pl

class Segmenter(pl.LightningModule):
    def __init__(self):
        super().__init__()
        import torch
        from model import ViViTBackbone
        self.model = ViViTBackbone(
            t=img_size[0],
            h=img_size[1],
            w=img_size[2],
            patch_t=8, #increases bias row size and decresease row of wiegth size
            patch_h=8,
            patch_w=8,
            num_classes=6,
            dim=128, # determines columns of bias size
            depth=6,#no influence
            heads=4,#no influence
            mlp_dim=8,#no influence
            model=3
        )
        self.train_loss = 0
        self.val_loss = 0
        # self.criterion = wandb.config['criterion']()#nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)

from model import ViViTBackbone

device = torch.device('cpu')
x = torch.rand(32, 3, 32, 64, 64).to(device)

vivit = ViViTBackbone(32, 64, 64, 8, 4, 4, 10, 512, 6, 10, 8, model=3).to(device)
out = vivit(x)
print(out)


# seg = Segmenter()

# print(sample.shape)
# out = seg(sample)
# print(out, out.shape)
