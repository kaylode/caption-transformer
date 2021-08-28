import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from .custom import CustomCutout
from configs import Config

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class Denormalize(object):
    """
    Denormalize image and boxes for visualization
    """
    def __init__(self, mean = MEAN, std = STD, **kwargs):
        self.mean = mean
        self.std = std
        
    def __call__(self, img, box = None, label = None, mask = None, **kwargs):
        """
        :param img: (tensor) image to be denormalized
        :param box: (list of tensor) bounding boxes to be denormalized, by multiplying them with image's width and heights. Format: (x,y,width,height)
        """
        mean = np.array(self.mean)
        std = np.array(self.std)
        img_show = img.numpy().squeeze().transpose((1,2,0))
        img_show = (img_show * std+mean)
        img_show = np.clip(img_show,0,1)
        return img_show

def get_resize_augmentation(image_size, keep_ratio=False, box_transforms = False):
    """
    Resize an image, support multi-scaling
    :param image_size: shape of image to resize
    :param keep_ratio: whether to keep image ratio
    :param box_transforms: whether to augment boxes
    :return: albumentation Compose
    """

    if not keep_ratio:
        return  A.Compose([
            A.Resize(
                height = image_size[1],
                width = image_size[0]
            )]) 
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=max(image_size)), 
            A.PadIfNeeded(min_height=image_size[1], min_width=image_size[0], p=1.0, border_mode=cv2.BORDER_CONSTANT),
            ])
        

def get_augmentation(_type='train'):

    config = Config('./augmentations/augments.yaml')
    flip_config = config.flip
    ssr_config = flip_config['shift_scale_crop']
    color_config = config.color
    quality_config = config.quality

    transforms_list = [

        A.HorizontalFlip(p=flip_config['hflip']),
    
        A.RandomBrightnessContrast(
            brightness_limit=color_config['brightness'], 
            contrast_limit=color_config['contrast'], 
            p=0.5),

        A.OneOf([
            A.IAASharpen(p=quality_config['sharpen']), 
            A.Compose([
                A.FromFloat(dtype='uint8', p=1),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=quality_config['clahe']),
                A.ToFloat(p=1),
            ])           
        ], p=quality_config['prob']),
        
        # A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
        A.ShiftScaleRotate(
            shift_limit=ssr_config['shift_limit'], 
            scale_limit=ssr_config['scale_limit'], 
            rotate_limit=ssr_config['rotate_limit'], 
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=ssr_config['prob']),
    ]
        
    transforms_list += [
        A.Normalize(mean=MEAN, std=STD, max_pixel_value=1.0, p=1.0),
        ToTensorV2(p=1.0)
    ]

    train_transforms = A.Compose(transforms_list)

    val_transforms = A.Compose([
        A.Normalize(mean=MEAN, std=STD, max_pixel_value=1.0, p=1.0),
        ToTensorV2(p=1.0)
    ])
    

    return train_transforms if _type == 'train' else val_transforms