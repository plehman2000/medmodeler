from data import MRIDATASET_PILLOW
from colorama import init, Fore, Back, Style
import matplotlib as mpl
import matplotlib
from pathlib import Path
import multiprocessing
import random
import enum
from IPython.display import display
import uuid
import cv2
import PIL
import torch.nn.functional as F
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from tqdm import tqdm
import torchio as tio
from PIL import Image
import torchvision.transforms as T
import logging
import os
import glob
import torch
from data import *
from model import *
import torchvision
import time

global most_recent_checkpoint 

''' download the checkpoint from the following link, or train a new model using "train.py"
 https://drive.google.com/file/d/1UXqUkiHx6kqwWfyMXWxv6MtU4h8OJpao/view?usp=share_link
'''
most_recent_checkpoint = os.getcwd() + "/checkpoints/checkpoint.ckpt"


t0 = time.time()

# Initializes Colorama
init(autoreset=True)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger()
logger.disabled = True


def rotate(image, radiological=True, n=-1):
    # Rotate for visualization purposes
    image = np.rot90(image, n)
    if radiological:
        image = np.fliplr(image)
    return image


def mask_to_im(arr):
    arr = torch.squeeze(arr, 0)
    arr = arr.permute(1, 2, 0).numpy()
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
            elif label == 4:
                new_arr[i][j] == [41, 120, 142]
            else:
                new_arr[i][j] = [0, 0, 0]
            # new_arr[i][j] = [0,150,0]
    del arr
    new_arr = new_arr.astype(np.uint8)
    new_arr = Image.fromarray(new_arr)
    return new_arr


def save_slice(image: Image):
    global SAMPLE_COUNT
    global LABEL_COUNT

    image = tio.ToCanonical()(image)  # type: ignore[assignment]
    data = image.data[-1]

    for i in tqdm(range(data.size()[2])):
        slice_z = rotate(data[:, :, i], radiological=True)
        arr = slice_z
        sm = matplotlib.cm.ScalarMappable(cmap=None)
        sm.set_clim(None, None)
        origin = mpl.rcParams["image.origin"]
        if (isinstance(arr, memoryview) and arr.format == "B"
                and arr.ndim == 3 and arr.shape[-1] == 4):
            # Such an ``arr`` would also be handled fine by sm.to_rgba (after
            # casting with asarray), but it is useful to special-case it
            # because that's what backend_agg passes, and can be in fact used
            # as is, saving a few operations.
            rgba = arr
        else:
            rgba = sm.to_rgba(arr, bytes=True)
        pil_shape = (rgba.shape[1], rgba.shape[0])
        image = PIL.Image.frombuffer(
            "RGBA", pil_shape, rgba, "raw", "RGBA", 0, 1)
        IMAGE_SLICES.append(image)


def save_subject_images(subject: tio.Subject):
    iterable = enumerate(subject.get_images_dict(intensity_only=False).items())
    for image_index, (name, image) in iterable:
        save_slice(image)


def tensor_to_label_map(tensor):
    arr = torch.squeeze(tensor, 0)
    arr = arr.permute(1, 2, 0).numpy()
    new_arr = np.zeros((len(arr[0]), len(arr[1])))
    lab = 0
    for i in range(len(arr[0])):
        for j in range(len(arr[1])):
            label = np.argmax(arr[i][j])
            # TODO CHECK IF THIS ELIMINATES SEGMENT FROM NIIGZ THAT COVERS THE ENTIRE BODY
            if arr[i][j][label] > 0.5 and label != 1:
                new_arr[i][j] = label
            else:
                new_arr[i][j] = 0

            # new_arr[i][j] = [0,150,0]
    del arr
    new_arr = new_arr.astype(np.uint8)
    new_arr = torch.tensor(new_arr)
    return new_arr


def inference(infilename, outfilename):
    DEVICE = torch.device('cpu')
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")

    global most_recent_checkpoint


    torch.manual_seed(0)
    architecture = "FPN"
    encoder_name = "resnet34"
    encoder_weights = "imagenet"
    model = MRISEG(architecture, encoder_name, encoder_weights='imagenet',  in_channels=3,
                   out_classes=5)
    model = model.load_from_checkpoint(most_recent_checkpoint)
    
    print(Style.BRIGHT + Back.CYAN + Fore.WHITE +
          f"Loaded Checkpoint at {most_recent_checkpoint}")


    model = model.to(DEVICE)


    test_file = infilename
    print(Back.GREEN + Fore.BLACK + f"Running inference on {test_file}")
    subject = tio.Subject(sample=tio.ScalarImage(test_file))


    global IMAGE_SLICES
    IMAGE_SLICES = []

    save_subject_images(subject)

    inference_dataset = []
    resize_trans = torchvision.transforms.Resize((512, 512))
    dataset = MRIDATASET_PILLOW(IMAGE_SLICES, transform=resize_trans)
    inference_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    inference_masks = []
    ct = 0

    for batch in tqdm(inference_dataloader):
        with torch.no_grad():
            model.eval()
            ret_dict = model.validation_step(batch["image"].to(DEVICE), 0)
            logits = ret_dict["mask"]
        pr_masks = logits.sigmoid()
        inference_masks.append(pr_masks)


    print(Style.BRIGHT + Back.CYAN + Fore.WHITE +
          "Converting inference results to .niigz file...")
    # ! TENSOR TO LABEL IS THE BIGGEST BOTTLENECK, CAN IT BE IMPROVED?
    masks = [tensor_to_label_map(x.to(torch.device('cpu')))
             for x in tqdm(inference_masks)]
    subject_tensor = torch.unsqueeze(torch.stack(masks, 2), 0)


    rotated_subject = torch.rot90(subject_tensor, 1, [1, 2])
    labels = tio.LabelMap(tensor=rotated_subject, affine=subject["sample"].affine)
    labels.save(outfilename)


    t1 = time.time()

    total = t1-t0
    print(Back.RED + Fore.BLACK +
          f"Total execution time for stack of {len(IMAGE_SLICES)}: {t1-t0} seconds")

    
    
inference(infilename="", outfilename="inference_output.nii.gz")