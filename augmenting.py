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
import PIL
from torchvision.utils import save_image
import os


dirs = os.listdir("./Final Handpicked")

for path in dirs:
    data_path = "./Final Handpicked/{}".format(path)
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])


    original_dataset = torchvision.datasets.ImageFolder(data_path, transform=transform)

    dataset = []
    for i in range(len(original_dataset)):
        
        # Transofrmations
        dataset.append(transforms.RandomHorizontalFlip(p=1)(original_dataset[i][0])) # Horizontal Flip
        dataset.append(transforms.RandomPerspective(distortion_scale=0.3, p=1)(original_dataset[i][0])) # Turns in 3d
        dataset.append(transforms.RandomRotation(degrees=(-90,90))(original_dataset[i][0])) # Rotates within given range
        dataset.append(transforms.ColorJitter(brightness = .7, hue = .2)(original_dataset[i][0])) # Changes color

        # Original image copy
        dataset.append(original_dataset[i][0])


    k = 0
    for i in range(len(dataset)):
        save_image(dataset[i], "./DatasetAugmented/{}/{}.jpg".format(path,k))
        k += 1

