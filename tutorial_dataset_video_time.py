# -*- coding: utf-8 -*-
# This script defines a dataset class for handling MRI data, including preprocessing.
# It also includes utility functions for time extraction, normalization, and batch sampling.

import json
import cv2
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
from torch.utils.data import Sampler
import numpy as np
from collections import defaultdict

import glob
import random
import os
import numpy as np
from os.path import splitext, isfile, join
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
from einops import rearrange, repeat
import pandas as pd

class ImageDataset(Dataset):
    def __init__(self, txt_path=r"E:\KJD\Codes\PyTorch-GAN-master\GAN-svn\train_0123.txt",
                 images_dir=r"E:\KJD\Data\16phases_reg1", resize_w=256, resize_h=256, istrain=True):
        print("Init Dataset")
        self.txt_path = txt_path
        self.patients_list = loadtxt(self.txt_path)
        self.images_dir = images_dir
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.npy_lists_contrastreg = []
        self.npy_lists_mask = []
        self.cls = []
        self.patients = []
        self.slices = []
        self.times = []
        self.indexes = []
        self.istrain = istrain

        """ Customize your dataset here. The current implementation reads an excel file to get the patient and time information"""
        excel_path = r"/media/volume/Data_in3_2/Data_in3_2_copy/MRICEKWorld/output.xlsx"
        df = pd.read_excel(excel_path)
        rows = df.shape[0]
        print('patient list', self.patients_list)
        for i in range(0, rows):#rows
            folder_name = df.iloc[i, 0] #patient
            print(folder_name)
            if folder_name not in self.patients_list :
                continue
            file_names = df.iloc[i, 1:].tolist()
            # Modify the fixed_path definition to dynamically find the folder containing 'mask' but not 'delay'
            fixed_path = None
            for subfolder in os.listdir(os.path.join(images_dir, folder_name)):
                if ('mask' in subfolder or 'pre' in subfolder) and 'delay' not in subfolder:
                    fixed_path = os.path.join(images_dir, folder_name, subfolder)
                    break
            if os.path.exists(fixed_path) == False:
                print('fixed path', fixed_path, os.path.exists(fixed_path))
            save_path = None
            cnt = 0
            for index in range(0, len(file_names[1:])): # time, from 1st artery series
                if txt_path.find("test") >=0 and index == 0:
                    continue
                j = file_names[1+index]
                if isinstance(j, str):
                    moving_path = os.path.join(images_dir, folder_name, j.replace('.nii.gz', ''))
                    save_path = os.path.join(images_dir, folder_name, j.replace('.nii.gz', ''))
                    if os.path.exists(save_path) == False:
                        print(save_path, "Moving Not exist")
                        continue
                    count = sum(1 for entry in os.scandir(str(fixed_path)) if
                                entry.is_file() and entry.name.lower().endswith('.npy'))
                    artery_1st_path = os.path.join(images_dir, folder_name, file_names[1].replace('.nii.gz', ''))
                    if os.path.exists(artery_1st_path) == False:
                        print(artery_1st_path, "1st Moving Not exist")
                        continue
                    time = calculate_time_difference(artery_1st_path, moving_path)
                    print(artery_1st_path, moving_path, time)
                    for slice in range(count):
                        mask_file = os.path.join(fixed_path, f"{os.path.basename(fixed_path)}_slice_{int(slice):02d}.npy")
                        art_file = os.path.join(moving_path, f"{j.replace('.nii.gz', '')}_slice_{int(slice):02d}.npy")

                        if os.path.exists(mask_file) == False or os.path.exists(art_file) == False:
                            print(mask_file, art_file, time, os.path.exists(mask_file), os.path.exists(art_file))
                        else:
                            self.npy_lists_mask.append(mask_file)
                            self.npy_lists_contrastreg.append(art_file)
                            self.cls.append(time)
                            self.slices.append(slice)
                            self.patients.append(folder_name)
                            self.indexes.append(index)
                            print('add', mask_file, art_file, time, slice, index, folder_name)



    def __getitem__(self, index):

        image_path, mask_path, time, patient, order = self.npy_lists_contrastreg[index], self.npy_lists_mask[index], self.cls[index], self.patients[index], self.indexes[index]
        slice = self.slices[index]
        diff, image, mean, std, mask, MIN, MAX = normalize_zeroscore2(mask_path, image_path, order)  # mask1&image用于求diff

        if self.istrain:
            transform_img = get_transform(self.resize_w, self.resize_h, 'train')
            transform_mask = get_transform(self.resize_w, self.resize_h, 'mask_train')
        else:
            transform_img = get_transform(self.resize_w, self.resize_h, 'val')
            transform_mask = get_transform(self.resize_w, self.resize_h, 'mask_val')

        seed = np.random.randint(2147483647)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        transformed_art = transform_img(image)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        transformed_plain = transform_mask(mask)
        transformed_diff = transform_mask(diff)

        transformed_plain = transformed_plain.cpu().numpy().astype(np.float32)
        transformed_diff = transformed_diff.cpu().numpy().astype(np.float32)
        transformed_art = transformed_art.cpu().numpy().astype(np.float32)

        three_plain = np.repeat(transformed_plain, 3, axis=0)
        three_diff = np.repeat(transformed_diff, 3, axis=0)
        three_art = np.repeat(transformed_art, 3, axis=0)

        canny_output = three_plain[2]
        three_plain = rearrange(three_plain, 'c h w -> h w c')
        three_diff = rearrange(three_diff, 'c h w -> h w c')
        three_art = rearrange(three_art, 'c h w -> h w c')

        diff = three_diff
        # print(diff.min(), diff.max())
        return dict(jpg=three_art, txt=str(time), hint=three_plain, art=three_art, min=MIN, max=MAX, mean=mean, std=std, slice = slice, patient = patient, order = order, canny=canny_output)

    def __len__(self):
        return len(self.npy_lists_mask)


def loadtxt(txt_path):
    with open(txt_path, 'r') as file:
        lines = file.readlines()
        # Remove newline characters from the lines
        lines = [line[:-1] for line in lines]  # 去掉换行符
        return lines

from datetime import datetime, timedelta

min_art_vals = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0 ]
max_art_vals = [941, 1023, 1098, 1167, 1170, 1220, 1356, 1353, 1349, 1350, 1348, 1339, 1202, 1153, 1194, 1127]
def normalize_zeroscore2(filename_mask, filename_art, order):
    mask_data = np.load(filename_mask)
    art_data = np.load(filename_art)

    normalized_mask = np.clip(mask_data, 0, 629)
    min_art_val = 0
    if order < 6: # Artery 
        max_art_vals = 1220
    elif order < 12: # Vein
        max_art_vals = 1356
    else: # Delay
        max_art_vals = 1202
    normalized_art = np.clip(art_data, min_art_val, max_art_vals)
    # print(filename_art, filename_mask, order, max_art_vals)
    diff = normalized_art - normalized_mask
    max_val = 1686
    min_val = -638
    diff = (diff - min_val) / (max_val - min_val)
    diff = diff * 2.0 - 1.0

    normalized_mask = (normalized_mask - 0) / 629.0
    normalized_mask = normalized_mask * 2.0 - 1.0
    normalized_art = (normalized_art - min_art_val) / (max_art_vals - min_art_val)
    normalized_art = normalized_art * 2.0 - 1.0

    return torch.tensor(diff).unsqueeze(dim=0), torch.tensor(normalized_art).unsqueeze(dim=0), 0, 1, torch.tensor(normalized_mask).unsqueeze(dim=0), 0, 0

def extract_time_from_filename(filename):
    # Extract the last two strings separated by underscores
    if len(filename.split('.nii')[0].split('_')[-1]) in [5,6]:
        time_str = filename.split('.nii')[0].split('_')[-1]
    else:
        time_str = filename.split('.nii')[0].split('_')[-2]
    # Determine the format based on the length of the time string
    if len(time_str) in [5, 6]:
        time_format = '%H%M%S'
    else:
        if len(filename.split('.nii.gz')[0].split('_')[-1]) in [5, 6]:
            time_str = filename.split('.nii.gz')[0].split('_')[-1]
        else:
            time_str = filename.split('.nii.gz')[0].split('_')[-2]
            if len(time_str) in [5, 6]:
                time_format = '%H%M%S'
            else:
                raise ValueError(f"Invalid time format in filename: {filename}")
    # Convert the time string to a datetime object
    return datetime.strptime(time_str, time_format)


def calculate_time_difference(filename1, filename2):
    # Extract and convert the times from the filenames
    time1 = extract_time_from_filename(filename1)
    time2 = extract_time_from_filename(filename2)
    # Calculate the time difference
    time_difference = abs(time1 - time2)
    return time_difference + timedelta(seconds=15)


def get_transform(osizew, osizeh, mode='train'):
    from torchvision.transforms import InterpolationMode
    if mode == 'train':
        transform_list = [
            # SquarePad(),
            transforms.Resize((osizew, osizeh), interpolation=InterpolationMode.BICUBIC),
            # transforms.RandomAffine(degrees=1,translate=[0.02, 0.02],scale=[0.98, 1.02]),

        ]

    elif mode == 'val':
        transform_list = [
            # SquarePad(),
            transforms.Resize((osizew, osizeh), interpolation=InterpolationMode.BICUBIC),

        ]
    elif mode == 'mask_train':
        transform_list = [
            # SquarePad(),
            transforms.Resize((osizew, osizeh), interpolation=InterpolationMode.NEAREST),
            # transforms.RandomAffine(degrees=1,translate=[0.02, 0.02],scale=[0.98, 1.02])
        ]
    elif mode == 'mask_val':
        transform_list = [
            # SquarePad(),
            transforms.Resize((osizew, osizeh), interpolation=InterpolationMode.NEAREST)]

    return transforms.Compose(transform_list)


class SquarePad:
    def __init__(self) -> None:
        pass

    def __call__(self, image):
        import copy
        img1 = copy.copy(image)
        img1 = img1.numpy()
        _, h, w = img1.shape
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)

        transform = transforms.Pad(padding, fill=0, padding_mode='constant')
        return transform(image)




class PatientSliceBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Build a mapping from (patient_id, slice_id) to a list of indices
        self.patient_slice_to_indices = {}
        for idx in range(len(dataset)):
            patient_id = dataset.patients[idx]
            slice_id = dataset.slices[idx]
            key = (patient_id, slice_id)
            self.patient_slice_to_indices.setdefault(key, []).append(idx)

        self.keys = list(self.patient_slice_to_indices.keys())

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.keys)

        for key in self.keys:
            indices = self.patient_slice_to_indices[key]
            if self.shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                yield indices[i:i + self.batch_size]

    def __len__(self):
        return sum((len(indices) + self.batch_size - 1) // self.batch_size
                   for indices in self.patient_slice_to_indices.values())