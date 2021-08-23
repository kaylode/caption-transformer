import os
import cv2
import torch
import numpy as np
import pandas as pd

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from augmentations.transforms import get_resize_augmentation, get_augmentation, MEAN, STD

from .utils import create_masks, make_text_feature_batch, split_patches

class ImageTextSet:
    """
    Input path to csv file contains texts
    """
    def __init__(self, input_path, csv_file, tokenizer, image_size=[512,512], keep_ratio=False, patch_size=16, type='train'):
        self.image_size = image_size
        self.patch_size = patch_size
        self.csv_file = csv_file
        self.input_path = input_path
        self.tokenizer = tokenizer
        self.transforms = A.Compose([
            get_resize_augmentation(image_size, keep_ratio=keep_ratio),
            get_augmentation(_type=type)
        ])
        self.load_data()

    def get_patch_dim(self):
        return self.patch_size * self.patch_size * 3
        
    def load_data(self):
        self.df = pd.read_csv(self.csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name, text = row.image_name, row.comment
        image_path = os.path.join(self.input_path, image_name)
        
        return {
            'image_path': image_path,
            'text': text,
        }

    def collate_fn(self, batch):
        
        image_paths = [s['image_path'] for s in batch]
        imgs = []
        ori_imgs = []
        image_names = []
        for image_path in image_paths:
            image_names.append(os.path.basename(image_path))
            ori_img = cv2.imread(image_path)
            image = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            ori_imgs.append(image.copy())
            image = image.astype(np.float32)
            image /= 255.0
            image = self.transforms(image=image)['image']
            imgs.append(image)
        imgs = torch.stack(imgs)

        imgs_patches = split_patches(imgs, self.image_size[0], self.image_size[1], P=16)
        image_masks = torch.ones(imgs_patches.shape[:-1])
        texts = [s['text'] for s in batch]
        
        tokens = self.tokenizer(texts, truncation=True)
        tokens = [np.array(i) for i in tokens['input_ids']]

        texts_ = make_text_feature_batch(
            tokens, pad_token=self.tokenizer.pad_token_id)
        
        texts_inp = texts_[:, :-1]
        texts_res = texts_[:, 1:]

        text_masks = create_masks(
            texts_inp,
            pad_token=self.tokenizer.pad_token_id, 
            is_tgt_masking=True)
        
        texts_inp = texts_inp.squeeze(-1)

        return {
            'image_names': image_names,
            'ori_imgs': ori_imgs,
            'image_patches': imgs_patches,
            'image_masks': image_masks.long(),
            'tgt_texts_raw': texts,
            'texts_inp': texts_inp.long(),
            'texts_res': texts_res.long(),
            'text_masks': text_masks.long(),
        }