# -*- coding: utf-8 -*-
"""Transfer_Learning_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Gm3f4oLpwWi4dQBIPaUYlG7WjkIQj_hQ

**Transfer Learning model using VGG**
"""

import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
import torchvision.models as models

# set up the random seed
torch.manual_seed(1)

!unzip 'DatasetAugmented_final' 
!unzip 'Validation Images (updated)'

vgg16 = models.vgg.vgg16(pretrained=True) #vgg16 setup

# 5 classes

classes = ['Black', 'Asian', 'Indian', 
           'White', 'MiddleEastern']

# Normalizes the dataset over all pixels
def mean_std_all(dataset):
    count = 0
    total = 0
    squared_total = 0
    
    for data in dataset:
        total += torch.sum(data[0])
        squared_total += torch.sum(data[0]**2)
        # print(data[0])
        count += 1
    mean = total / (count * 3 * 224 * 224)
    squared_mean = squared_total / (count * 3 * 224 * 224)
    std = (squared_mean - mean**2)**0.5
    return mean, std

# Normalizes the dataset overl 3 different RBG channels separately
def mean_std_seperate(dataset):
    total = [0,0,0]
    count = 0
    squared_total = [0,0,0]
    
    for data in dataset:
        for i in range(3):
            total[i] += torch.sum(data[0][i])
            squared_total[i] += torch.sum(data[0][i]**2)
        count += 1
    mean = [element / (count * 3 * 224 * 224) for element in total]
    squared_mean = [element / (count * 3 * 224 * 224) for element in squared_total]
    std = [(squared_mean[i] - mean[i]**2)**0.5 for i in range(3)]
    print(mean, std)
    return mean, std

def get_data(path, len_data):
    # ***** Specify the path to final dataset folder on your local machine ******
    data_path = path
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])

    # Seperate for training, validation, and test data
    dataset = torchvision.datasets.ImageFolder(data_path, transform=transform)
    num_data = math.floor(len(dataset) * len_data)
    print("Number of images: ", num_data)
    dummy_len = len(dataset) - (num_data)


    # Comment out till "Normalization" if you want to remove normalization code
    mean, std = mean_std_all(dataset)
    print("Mean: {}, Std: {}".format(mean, std))

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((mean),(std))])

    # plt.imshow(dataset[0][0].permute(1,2,0))
    # plt.show()

    dataset = torchvision.datasets.ImageFolder(data_path, transform=transform)
    # plt.imshow(dataset[0][0].permute(1,2,0))
    # plt.show()

    # Normalization

    # Split into train and validation
    data_set, dummy = torch.utils.data.random_split(dataset, [num_data, dummy_len])

    # return train_loader, val_loader, test_loader
    return data_set

class ANNClassifier(nn.Module):
    def __init__(self):
        super(ANNClassifier, self).__init__()
        self.name = "ANN"
        self.dropout = nn.Dropout(0.25)
        self.layer1 = nn.Linear(7 * 512 * 7, 5000) 
        self.layer1_n = nn.BatchNorm1d(5000)
        self.layer2 = nn.Linear(5000, 84)
        #self.layer2_n = nn.BatchNorm1d(84)
        self.layer3 = nn.Linear(84, 28)
        self.layer4 = nn.Linear(28, 5) #5 outputs


    def forward(self, img):
        flattened = img.view(-1, 7 * 512 * 7)

        activation1 = self.layer1(flattened)
        #activation1 = self.dropout(activation1)
        #activation1 = self.layer1_n(activation1)
        activation1 = F.relu(activation1)
        #activation1 = self.dropout(activation1)

        activation2 = self.layer2(activation1)

        #activation2 = self.layer2_n(activation2)

        activation2 = F.relu(activation2)

        activation3 = self.layer3(activation2)
        #activation3 = self.dropout(activation3)

        activation3 = F.relu(activation3)
        #activation3 = self.dropout(activation3)

        output = self.layer4(activation3)
        output = output.squeeze(1)
        return output

def get_accuracy(model, data, batch_size):
    correct = 0
    total = 0
    for imgs, labels in torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True):

        #############################################
        # To Enable GPU Usage
        if torch.cuda.is_available():
          imgs = imgs.cuda()
          labels = labels.cuda()
        #############################################

        output = model(VGGC(imgs))
        
        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]

        
        
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total

def save_the_model(new_val_acc, model):
    ''' If the new_val_acc beats the curr_acc, updates the best_model.
        Returns 1 if the best model was updated and 0 otherwise.
    '''

    if not os.path.exists("./best_model_acc.txt"): # Do necessary adjustments if the script is being run for the first time
        f = open("best_model_acc.txt", "w")
        f.write("0")
        f.close()
        curr_acc = 0
    else:
        f = open("best_model_acc.txt", "r")
        curr_acc = float(f.readline().strip())
        f.close()

    if curr_acc < new_val_acc:
        # Update the model
        print("Updating the model. Previous accuracy {} | New accuracy {}".format(curr_acc, new_val_acc))
        torch.save(model.state_dict(), "best_model")
        f = open("best_model_acc.txt", "w")
        f.write(str(new_val_acc))
        f.close()
        return 1
    return 0

def train(model, train_data, val_data, learning_rate=0.001, batch_size=64, num_epochs=1):
    torch.manual_seed(1000)  # set the random seed
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    iters, losses, train_acc, val_acc = [], [], [], []

    # Training
    start_time = time.time()
    n = 0 # the number of iterations
    for epoch in range(num_epochs):
        for imgs, labels in iter(train_loader):
            
            #############################################
            # To Enable GPU Usage
            if torch.cuda.is_available():
              imgs = imgs.cuda()
              labels = labels.cuda()
            #############################################
            
           

            out = model(VGGC(imgs))             # forward pass
            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch

            # Save the current training information
            iters.append(n)
            losses.append(float(loss)/batch_size)             # compute *average* loss
            n += 1


        train_acc.append(get_accuracy(model, train_data, batch_size)) # compute training accuracy 
        val_acc.append(get_accuracy(model, val_data, batch_size))  # compute validation accuracy
        print(("Epoch {}: Train acc: {} |"+"Validation acc: {}").format(
                epoch + 1,
                train_acc[-1],
                val_acc[-1]))
        

    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))

    # Plotting
    plt.title("Training Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()
    
    plt.title("Training Curve")

    plt.plot(range(1 ,num_epochs+1), train_acc, label="Train")
    plt.plot(range(1 ,num_epochs+1), val_acc, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))

print("Loading data sets...")
train_data = get_data("./DatasetAugmented", 1.0)
val_data = get_data("./Validation Images", 1.0)

model = ANNClassifier()
VGGC = vgg16.features

if torch.cuda.is_available():
    VGGC.cuda()
    model.cuda() 
    print('CUDA is available!  Training on GPU ...')
else:
    print('CUDA is not available.  Training on CPU ...')


torch.cuda.memory_summary(device=None, abbreviated=False)
train(model, train_data, val_data, 0.001, 16, 5)

train(model, train_data, val_data, 0.001, 16, 5)

train(model, train_data, val_data, 0.001, 32, 5)
