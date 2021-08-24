import torch
import os
import cv2
import torch
from tqdm import tqdm
import numpy as np

import albumentations as A
from augmentations.transforms import get_resize_augmentation, get_augmentation

from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader

class CocoDataset(Dataset):
    """
    Coco dataset
    """
    def __init__(self,root_dir, ann_path, image_size=[512,512], keep_ratio=False):

        self.root_dir = root_dir
        self.ann_path = ann_path
        self.image_size = image_size

        self.transforms = A.Compose([
            get_resize_augmentation(image_size, keep_ratio=keep_ratio),
            get_augmentation(_type='val')
        ])

        self.coco = COCO(ann_path)
        self.image_ids = self.coco.getImgIds()

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        return image_path

    def __getitem__(self, index):
        image_path = self.load_image(index)
        image_name = os.path.basename(image_path)
        image = self.load_augment(image_path)
        return image, image_name

    def load_augment(self, image_path):
        ori_img = cv2.imread(image_path)
        image = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image /= 255.0
        image = self.transforms(image=image)['image']
        return image

    def collate_fn(self, batch):
        imgs = [s[0] for s in batch]
        image_names = [s[1] for s in batch]

        return torch.stack(imgs), image_names


def split_patches(img, H, W, P):
    """
    Split image into patches. Apply for batches
    """
    assert int(W*H / (P*P)) * (P*P) == W*H, "Not divisible"
    num_patches_w = int(W/P)
    num_patches_h = int(H/P)
    num_channels = img.shape[0]

    patches = []
    for i in range(1, num_patches_h+1):
        for j in range(1, num_patches_w+1):
            patches.append(img[:, P*(i-1):P*i, P*(j-1):P*j].clone())
    patches = np.stack(patches).reshape(-1, num_channels*P*P)
        
    return patches  


def main():

    features_dir = "./flatten_features"
    dataset = CocoDataset(
        root_dir='./images',
        ann_path='./annotations/train.json',
        image_size=[384,384], keep_ratio=False)

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=2,
        pin_memory=False,
        collate_fn=dataset.collate_fn)

    for (imgs, image_names) in tqdm(dataloader):
        for img, image_name in zip(imgs, image_names):
            feats = split_patches(img, 384, 384, P=16)
            np.save(os.path.join(features_dir, image_name[:-4]), feats)

if __name__ == '__main__':
    main()