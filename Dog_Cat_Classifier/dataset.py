#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 20:36:09 2020

@author: Saphir Volviane
"""
from torch.utils.data import Dataset
import glob
import os
from PIL import Image


class DatasetLoader(Dataset):
    def __init__(self, path, transform=None):
        self.classes   = os.listdir(path)
        self.path      = [f"{path}/{className}" for className in self.classes]
        self.file_list = [glob.glob(f"{x}/*") for x in self.path]
        self.transform = transform
        
        files = []
        for i, className in enumerate(self.classes):
            for fileName in self.file_list[i]:
                files.append([i, className, fileName])
        self.file_list = files
        files = None
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fileName = self.file_list[idx][2]
        if self.file_list[idx][0]==2:
          classLabel = self.file_list[idx][0]-1
        else:
          classLabel = self.file_list[idx][0]
        image = Image.open(fileName)
        if self.transform:
            image = self.transform(image)
        return image, classLabel





