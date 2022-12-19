
import torchvision.transforms as T
from PIL import Image
import torchio as tio
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import torch.nn.functional as F
import torchio as tio
import torchvision
import PIL
import cv2
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
seed = 42

random.seed(seed)
torch.manual_seed(seed)


global SAMPLE_COUNT
global LABEL_COUNT
SAMPLE_COUNT = 0
LABEL_COUNT = 0
def rotate(image, radiological=True, n=-1):
    # Rotate for visualization purposes
    image = np.rot90(image, n)
    if radiological:
        image = np.fliplr(image)
    return image
def tensor_to_slices(inputs, repeat=1):
    slices = []
    for i in range(inputs.size()[3]):
        new = inputs[:, :, :, i]
        pil_image = transform(new)
        # display(pil_image)
        for _ in range(repeat):
            slices.append(pil_image)
    return slices

from tqdm import tqdm
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

    for frame in tqdm(frames):
        open_cv_image = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        writer.write(open_cv_image)

    writer.release()

import matplotlib as mpl
import matplotlib.pyplot as plt



def plot_volume(
        image: Image,
        radiological=True,
        channel=-1,  # default to foreground for binary maps
        axes=None,
        cmap=None,
        output_path=None,
        show=True,
        xlabels=True,
        percentiles=(0.5, 99.5),
        figsize=None,
        reorient=True,
        indices=None,
):
    global SAMPLE_COUNT
    global LABEL_COUNT
    """
    Image and label creation occurs here
    """
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=figsize)
    sag_axis, cor_axis, axi_axis = axes

    if reorient:
        image = tio.ToCanonical()(image)  # type: ignore[assignment]
    data = image.data[channel]


    # ! choose slices here
    
    kwargs = {}
    is_label = isinstance(image, tio.LabelMap)
    cmap = 'cubehelix' if is_label else 'gray'
    kwargs['cmap'] = cmap
    if is_label:
        kwargs['interpolation'] = 'none'

    sr, sa, ss = image.spacing
    kwargs['origin'] = 'lower'

    if percentiles is not None and not is_label:
        p1, p2 = np.percentile(data, percentiles)
        kwargs['vmin'] = p1
        kwargs['vmax'] = p2


    axi_aspect = sa / sr
    # * experimental saving code

    save_path = DATA_DIR + "//orange//org//"

    if is_label:
        save_path = save_path + "//lab//"
        print(data.size())
    else:
        save_path = save_path + "//im//"
        print(data.size())
              
    from tqdm import tqdm
    print("CALLED")
    for i in tqdm(range(data.size()[2])):
        slice_z = rotate(data[:, :, i], radiological=radiological)
        if is_label:
            LABEL_COUNT+=1
            temp_count = LABEL_COUNT
        else:
            SAMPLE_COUNT+=1
            temp_count = SAMPLE_COUNT
        plt.imsave(save_path + f'Sample{temp_count}.png', slice_z)
        

    return fig




def save_subject_images(
        subject: tio.Subject,
        cmap_dict=None,
        show=True,
        output_path=None,
        figsize=None,
        clear_axes=True,
        **kwargs,
):
    num_images = len(subject)
    many_images = num_images > 2
    subplots_kwargs = {'figsize': figsize}
    try:
        if clear_axes:
            subject.check_consistent_spatial_shape()
            subplots_kwargs['sharex'] = 'row' if many_images else 'col'
            subplots_kwargs['sharey'] = 'row' if many_images else 'col'
    except RuntimeError:  # different shapes in subject
        pass
    args = (3, num_images) if many_images else (num_images, 3)
    fig, axes = plt.subplots(*args, **subplots_kwargs)
    # The array of axes must be 2D so that it can be indexed correctly within
    # the plsot_volume() function
    axes = axes.T if many_images else axes.reshape(-1, 3)
    iterable = enumerate(subject.get_images_dict(intensity_only=False).items())
    axes_names = 'sagittal', 'coronal', 'axial'
    for image_index, (name, image) in iterable:
        image_axes = axes[image_index]
        cmap = None
        if cmap_dict is not None and name in cmap_dict:
            cmap = cmap_dict[name]
        last_row = image_index == len(axes) - 1
        plot_volume(
            image,
            axes=image_axes,
            show=False,
            cmap=cmap,
            xlabels=last_row,
            **kwargs,
        )

        




# import wandb

base_transforms = tio.Compose([tio.transforms.ToCanonical(),
                               tio.RescaleIntensity((0, 1))])


DATA_DIR = os.getcwd()


import regex as re




images_dir_PIL = DATA_DIR + "//orange//org//im//"
labels_dir_PIL  = DATA_DIR + "//orange//org//lab//"

# switched from augfix due to one hot error
dataset_dir = DATA_DIR + "//orange/org//" # .niigz dataset
images_dir = dataset_dir +  '//images'
labels_dir = dataset_dir + '//labels'
image_paths = [images_dir +  "//"+ x for x in os.listdir(images_dir) if x.endswith('.nii.gz')]
label_paths = [labels_dir +  "//"+x for x in os.listdir(labels_dir) if x.endswith('.nii.gz')]

image_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
label_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
assert len(image_paths) == len(label_paths)
dataset = []
i = 0
for (image_path, label_path) in tqdm(zip(image_paths, label_paths), total=len(image_paths)):
        print(image_path, label_path)
        norm = tio.Subject(
            sample=tio.ScalarImage(image_path),
            label=tio.LabelMap(label_path))
        save_subject_images(norm)
            
        
        # image_paths_PIL = [x for x in os.listdir(images_dir_PIL) if x.endswith('.png')]
        # label_paths_PIL = [x for x in os.listdir(labels_dir_PIL) if x.endswith('.png')]
        # image_paths_PIL.sort(key=lambda f: int(re.sub('\D', '', f)))
        # label_paths_PIL.sort(key=lambda f: int(re.sub('\D', '', f)))

        # sample_slices = [Image.open(str(images_dir_PIL) + p) for p in image_paths_PIL]
        # label_slices = [Image.open(str(labels_dir_PIL) + p) for p in label_paths_PIL]
        #SUPER FUCKING IMPORTANT: THIS IS NEEDED TO SAVE NEW VIDEO SAMPLES WHEN PROCESSING FILES IDEALLY
        # write_video(DATA_DIR + "//orange/video_data/samples/" + f"sample{i}.mp4", sample_slices, 30)
        # write_video(DATA_DIR + "//orange/video_data/labels/" + f"label{i}.mp4", label_slices, 30)

        # [os.remove(str(images_dir_PIL) + p) for p in image_paths_PIL]
        # [os.remove(str(labels_dir_PIL) + p) for p in label_paths_PIL]

#         i +=1
#         if i ==1:
#             break
    

