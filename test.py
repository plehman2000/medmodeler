from tqdm import tqdm
import time
import torchvision
from model import *
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
            elif label == 4:
                new_arr[i][j] == [41, 120, 142]
            else:
                new_arr[i][j] = [0, 0, 0]
    del arr
    new_arr = new_arr.astype(np.uint8)
    return new_arr


def display_samples(batch, pred_masks, ret_dict, ct, filename="all_samples"):
    # TODO EDIT TO WORK WITH BATCH
    for image, gt_mask, pr_mask in zip(batch["image"], batch["mask"], pred_masks):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        mask = mask_to_im(pred_masks[0].permute(1, 2, 0).numpy())
        pred_image = Image.fromarray(mask)
        pred_image.save('new_prediction.png')
        plt.imshow(pred_image)
        plt.title(f"Prediction")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        mask = batch["mask"]
        mask = torch.squeeze(mask, 0)
        mask = mask.permute(1, 2, 0).numpy()
        mask = mask_to_im(mask)
        im = Image.fromarray(mask)
        plt.imshow(im)
        plt.title("Mask")

        plt.axis("off")
        plt.savefig(filename + f"{ct}" + ".png")


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

logger = logging.getLogger()
logger.disabled = True


list_of_files = next(os.walk('./orange/MRI_VIT/'))[1]
most_recent_checkpoint_folder = max(list_of_files)
most_recent_checkpoint = os.getcwd() + r"/orange/MRI_VIT/" + most_recent_checkpoint_folder + "/checkpoints/" + \
    os.listdir(os.getcwd() + r"/orange/MRI_VIT/" +
               most_recent_checkpoint_folder + "/checkpoints/")[0]
print(most_recent_checkpoint)


torch.manual_seed(0)
root = str(pathlib.Path(__file__).parent.resolve())
images_directory = os.getcwd() + "/orange/org/"

test_dataset = MRIDATASET(images_directory, mode="test",
                          shuffle=False, clip_to=None)


print(f"Test size: {len(test_dataset)}")

n_cpu = os.cpu_count()
test_dataloader = DataLoader(
    test_dataset, batch_size=1, shuffle=True, num_workers=0)

architecture = "FPN"
encoder_name = "resnet34"
encoder_weights = "imagenet"
model = MRISEG(architecture, encoder_name, encoder_weights='imagenet',  in_channels=3,
               out_classes=5)


if most_recent_checkpoint != None:
    model = model.load_from_checkpoint(most_recent_checkpoint)


ct = 0

for batch in tqdm(test_dataloader):

    with torch.no_grad():
        model.eval()
        ret_dict = model.validation_step(batch['image'], 0)
        logits = ret_dict["mask"]

    pr_masks = logits.sigmoid()

    display_samples(batch, pr_masks, ret_dict, ct)
    ct += 1
    if ct == 10:
        break
