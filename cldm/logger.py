import os
import einops
import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only


class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, subdir="test", increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.subdir = subdir

    @rank_zero_only
    def inverse_zero_score_normalize(self, data, mean, var):
        return data * (np.sqrt(var)) + mean

    @rank_zero_only
    def zero_score_normalize(self, data, mean, var):
        normalized_img = (data - mean) * 1.0 / np.sqrt(var)
        return normalized_img

    @rank_zero_only
    def visualization_min_max_normalize(self, data):
        min_val = np.min(data)
        max_val = np.max(data)
        data_new = ((data - min_val) / (max_val - min_val))
        return data_new

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, self.subdir, split)
        # 获取batch size
        batch_size = None
        for k in images:
            if k not in ['mean', 'std', 'min', 'max', 'txt', 'slice', 'patient', 'order', 'canny', 'art']:
                if hasattr(images[k], 'shape') and len(images[k].shape) > 0:
                    batch_size = images[k].shape[0]
                    break
        
        if batch_size is None:
            batch_size = 1

        for batch_item_idx in range(batch_size):
            for k in images:
                if k.find("conditioning") >= 0 or k == 'mean' or k == 'std' or k == 'min' or k == 'max' or k == 'txt' or k == 'slice' or k == 'patient' or k == 'order':
                    continue
                if k.find("cfg") < 0 and self.subdir.find("continuous") > 0:
                    continue
                if k.find("reconstruction") >= 0 or k.find("canny") >= 0 or k.find("art") >= 0:
                    continue
                
                # Use temporary variable to avoid modifying original images[k]
                img_item = images[k]
                
                if k.find("hint") >= 0 : #plain
                    img_item = img_item.permute(0, 3, 1, 2)
                    img_item = img_item[batch_item_idx, 0, :, :].unsqueeze(0)
                    grid = img_item.cpu().numpy()
                    if self.rescale and split != "test":
                        grid = self.visualization_min_max_normalize(grid)
                        grid_new = (grid * 255).astype(np.uint8)
                    elif split == "test":
                        grid_new = grid

                if k.find("cfg") >= 0 : #predict
                    img_item = img_item[batch_item_idx, 0, :, :].unsqueeze(0)  
                    grid = img_item.cpu().numpy()
                    if self.rescale : #and split != "test"
                        grid_new = self.visualization_min_max_normalize(grid)
                        grid_new = (grid_new * 255).astype(np.uint8)
                    elif split == "test":
                        grid_new = grid
                    filename = "gs-{:06}_e-{:06}_b-{:06}_i-{:02d}_{}_txt={}_slice={:.2f}_patient={}_order={:.2f}.png".format(
                        global_step, current_epoch, batch_idx, batch_item_idx, 'predict', 
                        (images['txt'][batch_item_idx]).replace(":", "x"), 
                        float(images['slice'][batch_item_idx]), 
                        images['patient'][batch_item_idx], 
                        images['order'][batch_item_idx].item())
                    path = os.path.join(root, filename)
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                    if self.rescale: #and split != "test"
                        Image.fromarray(grid_new[0], mode='L').save(path)
                    if split == "test":
                        np.save(path.replace(".png", ".npy"), grid_new)
                    continue


                if k.find("art") >= 0:
                    img_item = img_item.permute(0, 3, 1, 2)
                    img_item = img_item[batch_item_idx, 0, :, :].unsqueeze(0)  
                    grid = img_item.cpu().numpy()
                    if self.rescale and split != "test":
                        grid_new = self.visualization_min_max_normalize(grid)
                        grid_new = (grid_new * 255).astype(np.uint8)
                    elif split == "test":
                        grid_new = grid

                if k.find("reconstruction") >= 0:
                    img_item = img_item[batch_item_idx, 0, :, :].unsqueeze(0) 
                    grid = img_item.cpu().numpy()
                    if self.rescale and split != "test":
                        grid_new = self.visualization_min_max_normalize(grid)#(grid+1.0)/2.0
                        grid_new = (grid_new * 255).astype(np.uint8)
                    elif split == "test":
                        grid_new = grid
                if k.find("canny") >= 0:
                    img_item = img_item[batch_item_idx, :, :].unsqueeze(0)  
                    grid = img_item.cpu().numpy()
                    if self.rescale and split != "test":
                        grid_new = self.visualization_min_max_normalize(grid)  # (grid+1.0)/2.0
                        grid_new = (grid_new * 255).astype(np.uint8)
                    elif split == "test":
                        grid_new = grid

                if k.find("jpg") >= 0 and self.subdir.find("continuous") < 0:
                    # mean var normalization -> [0, 1]
                    img_item = img_item.permute(0, 3, 1, 2)
                    img_item = img_item[batch_item_idx, 0, :, :].unsqueeze(0) 
                    grid = img_item.cpu().numpy()
                    if self.rescale and split != "test":
                        grid_new = self.visualization_min_max_normalize(grid)
                        grid_new = (grid_new * 255).astype(np.uint8)
                    elif split == "test":
                        grid_new = grid
                    filename = "gs-{:06}_e-{:06}_b-{:06}_i-{:02d}_{}_txt={}_slice={:.2f}_patient={}_order={:.2f}.png".format(
                        global_step, current_epoch, batch_idx, batch_item_idx, 'gt', 
                        (images['txt'][batch_item_idx]).replace(":", "x"), 
                        float(images['slice'][batch_item_idx]), 
                        images['patient'][batch_item_idx], 
                        images['order'][batch_item_idx].item())
                    path = os.path.join(root, filename)
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                    if self.rescale and split != "test":
                        Image.fromarray(grid_new[0], mode='L').save(path)
                    elif split == "test":
                        np.save(path.replace(".png", ".npy"), grid_new)

                    continue


                filename = "gs-{:06}_e-{:06}_b-{:06}_i-{:02d}_{}_txt={}_slice={:.2f}_patient={}_order={:.2f}.png".format(
                    global_step, current_epoch, batch_idx, batch_item_idx, k, 
                    (images['txt'][batch_item_idx]).replace(":", "x"), 
                    float(images['slice'][batch_item_idx]), 
                    images['patient'][batch_item_idx], 
                    images['order'][batch_item_idx].item())
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                if self.rescale and split != "test":
                    Image.fromarray(grid_new[0], mode='L').save(path)
                if split == "test":
                    np.save(path.replace(".png", ".npy"), grid_new)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                if k == 'txt' or k == 'patient':
                    images[k] = images[k]
                    continue
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    # if self.clamp:
                    #     images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="val")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="test")
