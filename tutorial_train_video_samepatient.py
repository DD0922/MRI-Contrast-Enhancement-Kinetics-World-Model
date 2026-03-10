import os.path
import sys

from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset_video_time import ImageDataset, PatientSliceBatchSampler
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.loggers import CSVLogger


from pytorch_lightning.callbacks import Callback
import torch

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 必须放在所有 CUDA 操作之前


print("start......")
# Configs
resume_path = r"E:\KJD\Codes\ControlNet-main\ControlNet\models\control_sd15_ini0215.ckpt"
if sys.platform.startswith("linux"):
    resume_path = r"/media/volume/Data_in3_2/Data_in3_2_copy/MRICEKWorld/control_sd15_ini0215.ckpt"


print('resume_path', resume_path)

batch_size = 4
if sys.platform.startswith("linux"):
    logger_freq = 5000
else:
    logger_freq = 5
learning_rate = 1e-5
sd_locked = False
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()

# model.load_state_dict(load_state_dict(resume_path, location='cpu'))
state_dict = load_state_dict(resume_path, location='cpu')
model.load_state_dict(state_dict, strict=False)


model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
train_txt = r"train1222.txt" # 
test_txt = r"test1222.txt"



if sys.platform.startswith("linux"):
    img_dir =  r"/media/volume/Data_in3_2/Data_in3_2_copy/Combine_Deformed_All/Image_Slices_All"
    nm = 16
dataset = ImageDataset(txt_path=train_txt,
                       images_dir=img_dir, resize_w=256, resize_h=256, istrain=True)
batch_sampler = PatientSliceBatchSampler(dataset, batch_size=4, shuffle=True)
train_dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=nm)
# train_dataloader = DataLoader(dataset, num_workers=nm)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=1,
            verbose=True,
            monitor="val/loss",
            mode="min",
            save_last=True,
            # every_n_train_steps=320700,
        )
csv_logger = CSVLogger(save_dir='logs/', name='MRI CEKWorld', version="1.1.8.76")
img_logger = ImageLogger(batch_frequency=logger_freq, clamp=False, subdir="image_log_1.1.8.76")


callbacks = [checkpoint_callback, img_logger]
trainer_params = {
    "gpus": 1,
    "max_epochs": 1,  # 1000
    # "max_steps": 320710,
    "logger": csv_logger,  # csvlogger
    # "tpu_cores":1,
    "num_sanity_val_steps": 1,  # 2
    "log_every_n_steps": 50,
    "check_val_every_n_epoch": 1,
    "num_sanity_val_steps": 5,
    "callbacks": callbacks,

}
trainer = pl.Trainer(**trainer_params)


# Train!
trainer.fit(model, train_dataloader)
