"""
Author: Honggu Liu

"""

from PIL import Image
from torch.utils.data import Dataset
import os
import random
import numpy as np
import cv2
import torch

class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []

        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)



class Dataset_CRNN(Dataset):
    def __init__(self, data_path, frame_list, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.frame_list = frame_list

        fh = open(data_path, 'r')
        videos = []

        for line in fh:
            line = line.rstrip()
            words = line.split()
            videos.append((words[0], int(words[1])))

        self.videos = videos


    def __getitem__(self, index):
        fn, label = self.videos[index]

        X = []

        for i in frame_list :
            fname = fn + "{}.jpg".format("{0:05d}".format(i))
            img = Image.open(fname).convert('RGB')
            if self.transform in not None:
                img = self.transform(img)
            X.append(img.squeze_(0))
        X = torch.stack(X, dim=0)
        print(X.shape)

        return X, label
