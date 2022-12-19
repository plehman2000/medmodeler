from data import mask_to_im
from datetime import datetime
import wandb
from pytorch_lightning.loggers import WandbLogger
import time
import torchvision
from model import *
from data import *
import torch
import glob
import os
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

logger = logging.getLogger()
logger.disabled = True


torch.manual_seed(0)


def display_samples(batch, pred_masks):
    now = datetime.now()
    temp = now.strftime("%H:%M:%S")
    time = temp.replace(":", "-")
    filename = f"all_samples_{time}.png"
    for image, gt_mask, pr_mask in zip(batch["image"], batch["mask"], pred_masks):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        mask = mask_to_im(pred_masks[0].permute(1, 2, 0).numpy())
        pred_image = Image.fromarray(mask)
        plt.imshow(pred_image)
        plt.title("Prediction")
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
        plt.savefig(filename)


root = str(pathlib.Path(__file__).parent.resolve())
images_directory = os.getcwd() + "/orange/org/"
resize_trans = torchvision.transforms.Resize(
    (512, 512))  # ! dimensions must be divisible by 32


# IMPORTANT: REDUCES SIZE OF TRAINING DATASET FOR TESTING PURPOSES e.g CLIP_TO = 10 means the datset is 10 images (not 10 .niigz files)
CLIP_TO = None

train_dataset = MRIDATASET(
    images_directory, mode="train", clip_to=CLIP_TO, transform=resize_trans)

print(train_dataset)

test_dataset = MRIDATASET(images_directory, mode="test",
                          clip_to=25, transform=resize_trans)


print(f"Train size: {len(train_dataset)}")
print(f"Test size: {len(test_dataset)}")

n_cpu = os.cpu_count()
# num_workers =between 2-8 * num GPU
train_dataloader = DataLoader(
    train_dataset, batch_size=8, shuffle=True, num_workers=8)
test_dataloader = DataLoader(
    test_dataset, batch_size=1, shuffle=False, num_workers=8)

architecture = "FPN"
encoder_name = "mit_b3"  # ! OG name"mit_b3"
encoder_weights = "imagenet"
global model
model = MRISEG(architecture, encoder_name,  in_channels=3, out_classes=5)

wandb.login()
wandb_logger = WandbLogger(project="MRI_VIT")
model.configure_optimizers(1e-5)
CHECKPOINT_PATH = os.getcwd() + ""  # Change this if you are training!


global batch
batch = next(iter(train_dataloader))


# TODO Add logging with weights and biases
trainer = pl.Trainer(
    default_root_dir=CHECKPOINT_PATH,
    accelerator='gpu',
    strategy='dp',
    devices=4,
    max_epochs=1,
    # profiler="simple",
    logger=wandb_logger,
    log_every_n_steps=2
)


CHECKPOINT_PATH = os.getcwd() + "/orange/ckpt/"
# !Train
trainer.fit(
    model,
    train_dataloader, ckpt_path=most_recent_checkpoint)


with torch.no_grad():
    model.eval()
    logits = model(batch["image"])


pr_masks = logits.sigmoid()
# print(f"data for mask:{data[:,:,0,0]}")
logging.info("prediction mask shape: %s", pr_masks.shape)
logging.info("ground truth mask shape: %s", batch["mask"].shape)
