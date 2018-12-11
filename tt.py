from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import os
from MobileNetV2 import *
import argparse
from data_prepare import ImageNetData
from data_prepare import *

parser = argparse.ArgumentParser(description="PyTorch implementation of SENet")
parser.add_argument('--data-dir', type=str, default="/disk/private-data/yy/CardMatching/new_data")
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--num-class', type=int, default=189)
parser.add_argument('--num-epochs', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.045)
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--gpus', type=str, default="0")
parser.add_argument('--print-freq', type=int, default=10)
parser.add_argument('--save-epoch-freq', type=int, default=1)
parser.add_argument('--save-path', type=str, default="/disk/private-data/yy/CardMatching/new_data/pytorch_output")
parser.add_argument('--resume', type=str, default="", help="For training from one checkpoint")
parser.add_argument('--start-epoch', type=int, default=0, help="Corresponding to the epoch of resume ")
args = parser.parse_args()
data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

a = ImageNetValDataSet(os.path.join(args.data_dir, 'list_val_3600.txt'),
                                               os.path.join(args.data_dir, 'photos_new'),
                                               data_transforms['val'])

