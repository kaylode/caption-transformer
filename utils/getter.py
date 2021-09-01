from metrics import *
from datasets import *
from models import *
from trainer import *
from augmentations import *
from loggers import *
from configs import *

import os
import cv2
import math
import json
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LambdaLR, ReduceLROnPlateau, OneCycleLR, CosineAnnealingWarmRestarts

from utils.utils import download_pretrained_weights
from utils.cuda import NativeScaler, get_devices_info

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from augmentations.transforms import MEAN, STD, get_resize_augmentation
from transformers import AutoTokenizer

from .random_seed import seed_everything

CACHE_DIR='./.cache'

def get_instance(config, **kwargs):
    # Inherited from https://github.com/vltanh/pytorch-template
    assert 'name' in config
    config.setdefault('args', {})
    if config['args'] is None:
        config['args'] = {}
    return globals()[config['name']](**config['args'], **kwargs)

def get_lr_policy(opt_config):
    optimizer_params = {}
    lr = opt_config['lr'] if 'lr' in opt_config.keys() else None
    if opt_config["name"] == 'sgd':
        optimizer = SGD
        optimizer_params = {
            'lr': lr, 
            'weight_decay': opt_config['weight_decay'],
            'momentum': opt_config['momentum'],
            'nesterov': True}
    elif opt_config["name"] == 'adam':
        optimizer = AdamW
        optimizer_params = {
            'lr': lr, 
            'eps': 1e-9,
            'weight_decay': opt_config['weight_decay'],
            'betas': (opt_config['momentum'], 0.98)}
    return optimizer, optimizer_params

def get_lr_scheduler(optimizer, lr_config, **kwargs):

    scheduler_name = lr_config["name"]
    step_per_epoch = False

    if scheduler_name == '1cycle-yolo':
        def one_cycle(y1=0.0, y2=1.0, steps=100):
            # lambda function for sinusoidal ramp from y1 to y2
            return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

        lf = one_cycle(1, 0.2, kwargs['num_epochs'])  # cosine 1->hyp['lrf']
        scheduler = LambdaLR(optimizer, lr_lambda=lf)
        step_per_epoch = True
        
    elif scheduler_name == '1cycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=0.001,
            epochs=kwargs['num_epochs'],
            steps_per_epoch=int(len(kwargs["trainset"]) / kwargs["batch_size"]),
            pct_start=0.1,
            anneal_strategy='cos', 
            final_div_factor=10**5)
        step_per_epoch = False
        

    elif scheduler_name == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=1,
            verbose=False, 
            threshold=0.0001,
            threshold_mode='abs',
            cooldown=0, 
            min_lr=1e-8,
            eps=1e-08
        )
        step_per_epoch = True

    elif scheduler_name == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs['num_epochs'],
            T_mult=1,
            eta_min=0.0001,
            last_epoch=-1,
            verbose=False
        )
        step_per_epoch = False

    elif scheduler_name == 'cosine2':
        scheduler = CosineWithRestarts(
            optimizer, 
            T_max=kwargs['train_len'])
        step_per_epoch = False
        
    return scheduler, step_per_epoch


def get_dataset_and_dataloader(config, bottom_up):

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    
    if not bottom_up:
        trainloader = EqualLengthTextLoader(
            ann_path=config.train_anns, root_dir=config.image_path,
            image_size=config.image_size, keep_ratio=config.keep_ratio,
            tokenizer=AutoTokenizer.from_pretrained(config.language),
            type='train', batch_size=config.batch_size, device=device)

        valloader = RawTextLoader(
            ann_path=config.val_anns, root_dir=config.image_path,
            image_size=config.image_size, keep_ratio=config.keep_ratio,
            tokenizer=AutoTokenizer.from_pretrained(config.language),
            type='val', batch_size=config.batch_size)
    else:
        trainloader = NumpyFeatureLoader(
            root_dir=config.image_path, npy_dir=config.npy_dir,
            batch_size=config.batch_size,
            ann_path=config.train_anns, device=device,
            tokenizer=AutoTokenizer.from_pretrained(config.language))

        valloader = RawNumpyFeatureLoader(
            root_dir=config.image_path, npy_dir=config.npy_dir,
            batch_size=32,
            ann_path=config.val_anns,
            tokenizer=AutoTokenizer.from_pretrained(config.language))

    return  trainloader.dataset, valloader.dataset, trainloader, valloader

import torch
import numpy as np
# code from AllenNLP

class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with restarts.
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    T_max : int
        The maximum number of iterations within the first cycle.
    eta_min : float, optional (default: 0)
        The minimum learning rate.
    last_epoch : int, optional (default: -1)
        The index of the last epoch.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 T_max: int,
                 eta_min: float = 0.,
                 last_epoch: int = -1,
                 factor: float = 1.) -> None:
        # pylint: disable=invalid-name
        self.T_max = T_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart: int = 0
        self._cycle_counter: int = 0
        self._cycle_factor: float = 1.
        self._updated_cycle_len: int = T_max
        self._initialized: bool = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time get_lr() was called, since
        # we want to start with step = 0, but _LRScheduler calls get_lr with
        # last_epoch + 1 when initialized.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lrs = [
            (
                self.eta_min + ((lr - self.eta_min) / 2) *
                (
                    np.cos(
                        np.pi *
                        ((self._cycle_counter) % self._updated_cycle_len) /
                        self._updated_cycle_len
                    ) + 1
                )
            ) for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.T_max)
            self._last_restart = step

        return lrs