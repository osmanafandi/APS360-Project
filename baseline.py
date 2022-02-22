import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

torch.manual_seed(1)  # set the random seed

class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.name = "Baseline"
        self.layer1 = nn.Linear(3 * 224 * 224, 84) #images are rgb 224 x 224 pixels
        self.layer2 = nn.Linear(84, 28)
        self.layer3 = nn.Linear(28, 7) #7 outputs
    def forward(self, img):
        flattened = img.view(-1, 3 * 224 * 224)
        activation1 = self.layer1(flattened)
        activation1 = F.relu(activation1)
        activation2 = self.layer2(activation1)
        activation2 = F.relu(activation2)
        output = self.layer3(activation2)
        output = output.squeeze(1)
        return output

#TRAIN BASELINE ...

#For viewing multiclass probabilities!
prob = F.softmax(output, dim=1)
print(prob)
print(sum(prob[0]))