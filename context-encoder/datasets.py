import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, img_size=128, mask_size=64, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.img_size = img_size
        self.mask_size = mask_size
        self.mode = mode
        self.files = sorted(glob.glob("%s/*.png" % root))
        self.files = self.files[:-100] if mode == "train" else self.files[-100:]

        self.mouth_positions = dict()

        with open("../mouth_positions.txt", "r") as f:
            for line in f:
                # filename lefteye_x lefteye_y righteye_x righteye_y nose_x nose_y leftmouth_x leftmouth_y rightmouth_x rightmouth_y
                s = line.strip().split()
                filename = s[0][:-4]
                leftmouth_x = int(s[-4])
                leftmouth_y = int(s[-3])
                rightmouth_x = int(s[-2])
                rightmouth_y = int(s[-1])
                # print("{}/{}.png".format(root, filename))
                self.mouth_positions["{}/{}.png".format(root, filename)] = (leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)

    def apply_random_mask(self, img):
        """Randomly masks image"""
        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1

        return masked_img, masked_part

    def apply_center_mask(self, img):
        """Mask center part of image"""
        # Get upper-left pixel coordinate
        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, i : i + self.mask_size, i : i + self.mask_size] = 1

        return masked_img, i

    def apply_mustache_mask(self, img, filename):
        """Mask mustache of image"""
        leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y = self.mouth_positions[filename]
        masked_part = img[:, leftmouth_y:rightmouth_y, leftmouth_x:rightmouth_x]
        masked_img = img.clone()
        masked_img[:, leftmouth_y:rightmouth_y, leftmouth_x:rightmouth_x] = 1

        return masked_img, masked_part

    def __getitem__(self, index):
        filename = self.files[index % len(self.files)]
        img = Image.open(filename)
        img = self.transform(img)
        if self.mode == "train":
            # For training data perform random mask
            masked_img, aux = self.apply_random_mask(img)
        else:
            # For test data mask the center of the image
            masked_img, aux = self.apply_center_mask(img)

            # For test data mask the mustache part
            # masked_img, aux = self.apply_mustache_mask(img, filename)

        return img, masked_img, aux

    def __len__(self):
        return len(self.files)
