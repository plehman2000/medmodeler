
import torch
from model import ViT


''' NOTE I have altered the model.py code to return the latent vector,
which will be used to construct the label video
'''
v = ViT(
    image_size = 128,          # image size
    frames = 16,               # number of frames
    image_patch_size = 16,     # image patch size
    frame_patch_size = 2,      # frame patch size
    num_classes = 1000,
    dim = 1024,                #controls size of latent vector, important parameter 
    spatial_depth = 6,         # depth of the spatial transformer
    temporal_depth = 6,        # depth of the temporal transformer
    heads = 8,
    mlp_dim = 2048
)
#
video = torch.randn(1, 3, 16, 128, 128) # (batch, channels, frames, height, width)

preds = v(video) # (4, 1000)
print(f"Pred Shape: {preds.shape}")
print(preds)