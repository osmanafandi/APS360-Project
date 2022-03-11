import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math

class CNN_Model(nn.Module):
    #Should decide on default value for num_classes
    def __init__(self, in_channels = 3, num_classes=7):
        super(CNN_Model, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),


        )

        self.linear =  nn.sequential(
            nn.Linear(in_features=256*14*14, out_features=4096),
            nn.Relu(),
            #Could potentially add dropout here
            nn.Linear(in_features=4096,out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096,out_features=self.num_classes)
        )

    def forward(self,img):
        img = self.conv(img)
        img = img.view(img.size(0),-1)
        img = self.linear(img)
        return img



