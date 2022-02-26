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

torch.manual_seed(1000)  # set the random seed

def get_data(len_train_data, len_val_data, len_test_data):

    # ***** Specify the path to final dataset folder on your loca machine ******
    data_path = "./Final Dataset"
    transform = transforms.Compose([transforms.Resize((224,224)), 
                                    transforms.ToTensor()])

    #Seperate for training, validation, and test data 
    dataset = torchvision.datasets.ImageFolder(data_path, transform=transform)
    num_train = math.floor(len(dataset)*len_train_data)
    num_val = math.floor(len(dataset)*len_val_data)
    num_test = math.floor(len(dataset)*len_test_data)
    print("Number of images for training: ", num_train)
    print("Number of images for validation: ", num_val)
    print(num_test)
    dummy_len = len(dataset) - (num_test + num_train + num_val)

    # Split into train and validation
    train_set, val_set, test_set, dummy = torch.utils.data.random_split(dataset, [num_train, num_val, num_test, dummy_len]) #80%, 10%, 10% split


    # return train_loader, val_loader, test_loader
    return train_set, val_set, test_set

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

def get_accuracy(model, data):
    correct = 0
    total = 0
    for imgs, labels in torch.utils.data.DataLoader(data, batch_size=64):

        #############################################
        # To Enable GPU Usage
        if torch.cuda.is_available():
          imgs = imgs.cuda()
          labels = labels.cuda()
        #############################################
        
        output = model(imgs)
        
        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total

def train(model, train_data, val_data, learning_rate=0.001, batch_size=64, num_epochs=1):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

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
            
              
            out = model(imgs)             # forward pass
            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch

            # Save the current training information
            iters.append(n)
            losses.append(float(loss)/batch_size)             # compute *average* loss
            n += 1
        train_acc.append(get_accuracy(model, train_data)) # compute training accuracy 
        val_acc.append(get_accuracy(model, val_data))  # compute validation accuracy
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
train_data, val_data, test_data = get_data(0.08, 0.02, 0.02)

baseline_model = Baseline()
print("Training baseline...")
if torch.cuda.is_available():
    baseline_model.cuda() #USE GPU!

train(baseline_model, train_data, val_data, 0.001, 64, 30)