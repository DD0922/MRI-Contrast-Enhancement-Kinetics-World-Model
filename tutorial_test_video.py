import os.path
import sys

from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset_video_time import ImageDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.loggers import CSVLogger
from transformers import CLIPTokenizer

if sys.platform.startswith("linux"):
    resume_path = r"/media/volume/Data_in3_2/Data_in3_2_copy/MRICEKWorld/v76_last.ckpt"

batch_size = 16
if sys.platform.startswith("linux"):
    logger_freq = 1
else:
    logger_freq = 1
learning_rate = 1e-5
sd_locked = False

only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
state_dict = load_state_dict(resume_path, location='cpu')
model.load_state_dict(state_dict, strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control



train_txt = r"train1222.txt" #
test_txt = r"test1222.txt"



if sys.platform.startswith("linux"):
    img_dir =  r"/media/volume/Data_in3_2/Data_in3_2_copy/Combine_Deformed_All/Image_Slices_All"
    nm = 16

test_dataset = ImageDataset(txt_path=test_txt,
                           images_dir=img_dir, resize_w=256, resize_h=256, istrain=False)
test_dataloader = DataLoader(test_dataset, num_workers=nm, batch_size=batch_size, shuffle=False)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=1,
            verbose=True,
            monitor="val/loss",
            mode="min",
            save_last=True,
        )
csv_logger = CSVLogger(save_dir='logs/', name='MRI CEKWorld', version="v_1.1.8.76_test")
# img_logger = ImageLogger(batch_frequency=logger_freq, clamp=False, subdir="image_log_1.1.8.82/predictall_order_ep4")
img_logger = ImageLogger(batch_frequency=logger_freq, max_images=batch_size, clamp=False, subdir="image_log_1.1.8.76/predict", log_images_kwargs={'N': batch_size} )



callbacks = [checkpoint_callback, img_logger]
trainer_params = {
    "gpus": 1,
    "max_epochs": 100,  # 1000
    "logger": csv_logger,  # csvlogger
    # "tpu_cores":1,
    "num_sanity_val_steps": 5,  # 2
    "log_every_n_steps": 50,
    "check_val_every_n_epoch": 1,
    "num_sanity_val_steps": 5,
    "callbacks": callbacks,

}
trainer = pl.Trainer(**trainer_params)


#Test
trainer.test(model, dataloaders=test_dataloader)