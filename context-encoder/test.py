import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch
from datasets import *
from models import *

import torch.nn as nn
import torch.nn.functional as F
import torch


cuda = True if torch.cuda.is_available() else False

generator = torch.load('generator.pth')


transforms_ = [
    transforms.Resize((128, 128), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

test_dataloader = DataLoader(
    ImageDataset("test_image", transforms_=transforms_, mode="val"),
    batch_size=3,
    shuffle=True,
    num_workers=1,
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



samples, masked_samples, i = next(iter(test_dataloader))
samples = Variable(samples.type(Tensor))
masked_samples = Variable(masked_samples.type(Tensor))
i = i[0].item()  # Upper-left coordinate of mask
    # Generate inpainted image
gen_mask = generator(masked_samples)
filled_samples = masked_samples.clone()
filled_samples[:, :, i : i + 64, i : i + 64] = gen_mask
    # Save sample
sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -2)
save_image(sample, "output.png", nrow=6, normalize=True)