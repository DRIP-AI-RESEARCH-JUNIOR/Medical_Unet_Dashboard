import os
import random
from PIL import Image
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms

class SyntheticCellDataset(Dataset):

    def __init__(
        self,
        root
    ):
        # self.image_list = glob.glob(image_dir + '\*.TIF')
        # self.mask_list = glob.glob(mask_dir + '\*.TIF')
        mask_lst_ref = []
        self.root = root
        self.image_list = sorted(os.listdir(os.path.join(root, 'image')), key=len)
        self.mask_list = sorted(os.listdir(os.path.join(root, 'mask')), key=len)
        # for img in glob.glob(image_dir + '/*.TIF'):
        #     self.mask_list.append(img)
        #
        # for img in glob.glob(mask_dir + '/*.TIF'):
        #     self.image_list.append(img)

        # self.image_list.sort()
        # self.mask_list.sort()

    def transform(self, image, mask):
        # Resize
        resize = transforms.Resize(size=(576, 576))
        # print('asila')


        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        image = resize(image)
        mask = resize(mask)
        return image, mask

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.root, 'image',self.image_list[idx]))
        image = Image.fromarray(image).convert('L')

        #image = image.convert('L')
        mask = cv2.imread(os.path.join(self.root, 'mask', self.mask_list[idx]))
        mask = Image.fromarray(mask).convert('L')
        #mask = mask.convert('L')

        x, y = self.transform(image, mask)
        # return tensors
        return x, y
