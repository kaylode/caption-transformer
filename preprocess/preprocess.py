import argparse
import os
import torch
import tqdm
import numpy as np
from PIL import Image

from pycocotools.coco import COCO

from torchvision import transforms
from .inception import inception_v3_base

parser = argparse.ArgumentParser('Training Object Detection')
parser.add_argument('--features_dir', type=str, help='Feature output dir')

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, ann_path):
        self.coco = COCO(ann_path)
        self.image_ids = self.coco.getImgIds()
        self.root_dir = root_dir

        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        return image_path

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_path = self.load_image(idx)
        with Image.open(image_path).convert('RGB') as img:
            return self.transform(img), self.image_ids[idx]



def main(args):
    
    device = torch.device('cuda')
    inception = inception_v3_base(pretrained=True)
    inception.eval()
    inception.to(device)

    features_dir = args.features_dir

    dataset = ImageDataset(
        root_dir='/content/data/',
        ann_path='train.json')

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=32,
                                         num_workers=2,
                                         pin_memory=True,
                                         shuffle=False)

    with torch.no_grad():
        for imgs, ids in tqdm.tqdm(loader):
            outs = inception(imgs.to(device)).permute(0, 2, 3, 1).view(-1, 64, 2048)
            for out, id in zip(outs, ids):
                out = out.cpu().numpy()
                id = str(id.item())
                np.save(os.path.join(features_dir, id), out)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)