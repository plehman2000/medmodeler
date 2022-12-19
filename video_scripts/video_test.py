import os
import regex as re
test_videos_directory = os.getcwd() + "/orange/video_data/"
video_files = os.listdir(test_videos_directory)

video_file = test_videos_directory + "/samples/sample10.mp4"

import shutil

shutil.rmtree(test_videos_directory + "/generated_masks/")
os.mkdir(test_videos_directory + "/generated_masks/")

shutil.rmtree(test_videos_directory + "/temp_images/")
os.mkdir(test_videos_directory + "/temp_images/")


from PIL import Image


import time
import torchvision

from data import *
import torch
import glob
import os
import logging


def mask_to_im(arr):
    new_arr = np.zeros((len(arr[0]), len(arr[1]), 3))
    lab = 0
    for i in range(len(arr[0])):
        for j in range(len(arr[1])):
            label = np.argmax(arr[i][j])
            if label == 0:
                new_arr[i][j] = [34, 167, 132]
            elif label == 1:
                new_arr[i][j] = [68, 1, 84]
            elif label == 2:
                new_arr[i][j] = [76, 231, 255]
            elif label == 3:
                new_arr[i][j] = [64, 67, 135]
            elif label ==4:
                new_arr[i][j] == [41, 120, 142]
            else:
                new_arr[i][j] = [0, 0, 0]
            # new_arr[i][j] = [0,150,0]
    del arr
    new_arr = new_arr.astype(np.uint8)
    return new_arr



def to_PIL(pred_masks):
    mask = mask_to_im(pred_masks[0].permute(1, 2, 0).numpy())
    pred_image = Image.fromarray(mask)
    return pred_image
    # pred_image.save(filename)


# * should return a tensor in CHW orientation
def read_images(image_path):
    sample_image = Image.open(image_path)
    image = torch.tensor(np.array(sample_image.convert("RGB")).transpose(2,0,1)) # TODO should probably be altered to only have one conversion 
    return image


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
    
    
        
        
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

logger = logging.getLogger()
logger.disabled = True



resize_trans = torchvision.transforms.Resize((512,512)) 




    
    
    
most_recent_checkpoint = os.getcwd() + r"/orange/MRI_VIT/8iswa4he/checkpoints/epoch=0-step=6291.ckpt"
print(f"CHECKPOINT: {most_recent_checkpoint}")


sample_dir = os.getcwd() + "/orange/video_data/temp_images/"
images = []
frame_files = []


import cv2
vidcap = cv2.VideoCapture(video_file)
success,image = vidcap.read()
count = 0
while success:
    cv2.imwrite(f"{sample_dir}/{count}.png", image)     # save frame as JPEG file      
    success,image = vidcap.read()
    frame_files.append(image)
    count += 1
    


file_count = len(frame_files)
frame_files = [f"{x}.png" for x in range(file_count-1)]
i = 0
for image in frame_files:
    images.append(read_images(sample_dir + f"/{image}"))
    i+=1
    

    
from tqdm import tqdm


from model import *
model = MRISEG("FPN", "resnet34", encoder_weights='imagenet',  in_channels=3, out_classes=5)

if most_recent_checkpoint != None:
    model = model.load_from_checkpoint(most_recent_checkpoint)

filename = os.getcwd() + r"/orange/video_data/generated_masks/"
ct = 0
pred_images = []
device = torch.device("cuda")

model = model.to(device)
for image in tqdm(images):
    with torch.no_grad():
        model.eval()
        image = image.to(device)
        logits = model.forward(image)
    pr_masks = logits.sigmoid().cpu()
    im = to_PIL(pr_masks)
    pred_images.append(im)
    ct +=1
    

write_video(os.getcwd() + f"/orange/video_data/video/samplecount{ct}.mp4",pred_images, 10)
