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
            else:
                new_arr[i][j] = [0, 0, 0]
            # new_arr[i][j] = [0,150,0]
    del arr
    new_arr = new_arr.astype(np.uint8)
    return new_arr



def display_samples(batch, pred_masks, filename="all_samples"):
    ct = 0
    #TODO EDIT TO WORK WITH BATCH_SIZE > 1
    for image, gt_mask, pr_mask in zip(batch["image"], batch["mask"], pred_masks):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
        plt.title("Image")
        plt.axis("off")



        plt.subplot(1, 3, 3)
        mask = batch["mask"]
        mask = torch.squeeze(mask, 0)
        print(f"mask shape: {mask.shape}")
        mask = mask.permute(1, 2, 0).numpy()
        mask = mask_to_im(mask)
        im = Image.fromarray(mask)
        # im.save('test.png')
        plt.imshow(im)
        plt.title("Mask")

        plt.axis("off")
        plt.savefig(filename + f"{ct}" + ".png")
        ct += 1

        
        
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

logger = logging.getLogger()
logger.disabled = True




list_of_files = next(os.walk('./lightning_logs/'))[1]
most_recent_checkpoint_folder = max(list_of_files)
most_recent_checkpoint = os.getcwd() + r"/lightning_logs/" + most_recent_checkpoint_folder + "/checkpoints/" + \
    os.listdir(os.getcwd() + r"/lightning_logs/" +
               most_recent_checkpoint_folder + "/checkpoints/")[0]
print(most_recent_checkpoint)


torch.manual_seed(0)
root = str(pathlib.Path(__file__).parent.resolve())
images_directory = os.getcwd() + "/orange/org/"

# , transform=resize_trans)
test_dataset = MRIDATASET(images_directory, mode="test")


print(f"Test size: {len(test_dataset)}")

n_cpu = os.cpu_count()
test_dataloader = DataLoader(
    test_dataset, batch_size=1, shuffle=True, num_workers=0)

architecture = "FPN"
encoder_name = "resnet34"  # ! OG name"mit_b3"
encoder_weights = "imagenet"  # ? test how well these weights work
model = MRISEG(architecture, encoder_name, encoder_weights='imagenet',  in_channels=3,
               out_classes=4)  # ? TODO ? Convert to raw parameters (encoder depth, width, etc)


# ! for testing
# most_recent_checkpoint = None

if most_recent_checkpoint != None:
    model = model.load_from_checkpoint(most_recent_checkpoint)


batch = next(iter(test_dataloader))



pr_masks = []

# print(pr_masks.shape, pr_masks[0,:,0,0])


display_samples(batch, pr_masks)
