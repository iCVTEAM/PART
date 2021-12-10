import os
from torch.backends import cudnn

from config import Config
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger
import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
from vis import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='/media/space/ZYF/Dataset/Other/NAbirds/select/',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


if __name__ == "__main__":
    args = get_args()
    svdir = './vis'
    for filename in os.listdir(args.image_path):
        print (filename)

        process_cam(filename)
