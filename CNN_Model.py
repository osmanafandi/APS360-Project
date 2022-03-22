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
    def __init__(self, in_channels = 3, num_classes=6):
        super(CNN_Model, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv = nn.Sequential(
            #nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )

        self.linear =  nn.Sequential(
            # nn.BatchNorm1d(512*7*7),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(),
            # nn.BatchNorm1d(4096),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=4096,out_features=4096),
            nn.ReLU(),
            # nn.BatchNorm1d(4096),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=4096,out_features=self.num_classes)
        )

    def forward(self,img):
        img = self.conv(img)
        img = img.view(img.size(0),-1)
        img = self.linear(img)
        return img


#TRAINING CODE FOR THE CNN_MODEL

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


def get_data(len_train_data, len_val_data, len_test_data):
    # ***** Specify the path to final dataset folder on your loca machine ******
    data_path = "./DatasetAugmented"
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])

    # Seperate for training, validation, and test data
    dataset = torchvision.datasets.ImageFolder(data_path, transform=transform)
    num_train = math.floor(len(dataset) * len_train_data)
    num_val = math.floor(len(dataset) * len_val_data)
    num_test = math.floor(len(dataset) * len_test_data)
    print("Number of images for training: ", num_train)
    print("Number of images for validation: ", num_val)
    print(num_test)
    dummy_len = len(dataset) - (num_test + num_train + num_val)


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
    train_set, val_set, test_set, dummy = torch.utils.data.random_split(dataset, [num_train, num_val, num_test,
                                                                                  dummy_len])  # 80%, 10%, 10% split

    # return train_loader, val_loader, test_loader
    return train_set, val_set, test_set

def get_accuracy(model, data_loader, batch_size):
    correct = 0
    total = 0
    for imgs, labels in data_loader:

        #############################################
        # To Enable GPU Usage
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        #############################################

        output = model(imgs)

        # select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
        
    return correct / total

#list_of_classes = ['AsianAugmented', 'BlackAugmented', 'IndianAugmented', 'LatinoAugmented', 'MiddleEasternAugmented', 'WhiteAugmented']

def train(model, train_data, val_data, learning_rate=0.001, batch_size=64, num_epochs=1):
    torch.manual_seed(1000)  # set the random seed
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    iters, losses, train_acc, val_acc = [], [], [], []

    # Training
    start_time = time.time()
    n = 0  # the number of iterations
    for epoch in range(num_epochs):
        for imgs, labels in iter(train_loader):

            #############################################
            # To Enable GPU Usage
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            #############################################

            out = model(imgs)  # forward pass
            loss = criterion(out, labels)  # compute the total loss
            loss.backward()  # backward pass (compute parameter updates)
            optimizer.step()  # make the updates for each parameter
            optimizer.zero_grad()  # a clean up step for PyTorch

            # Save the current training information
            iters.append(n)
            losses.append(float(loss) / batch_size)  # compute *average* loss
            n += 1

        train_acc.append(get_accuracy(model, train_loader, batch_size))  # compute training accuracy
        val_acc.append(get_accuracy(model, val_loader, batch_size))  # compute validation accuracy
        print(("Epoch {}: Train acc: {} |" + "Validation acc: {}").format(
            epoch + 1,
            train_acc[-1],
            val_acc[-1]))
        # preds = torch.max(output.data, 1)
        # acc = [0 for c in list_of_classes]
        # for c in list_of_classes:
        #     acc[c] = (((preds == labels) * (labels == c)).float() / (max(labels == c).sum(), 1))
        #     print("Accuracy for ", c, " is ", acc[c])
        
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
    plt.plot(range(1, num_epochs + 1), train_acc, label="Train")
    plt.plot(range(1, num_epochs + 1), val_acc, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))

# cnn_model = CNN_Model()
# if torch.cuda.is_available():
#     cnn_model.cuda() #USE GPU!

# print("Testing for overfit (sanity check)...")
# train_data, val_data, test_data = get_data(0.1, 0.1, 0) #load small dataset for overfit test
# train(cnn_model, train_data, val_data, 0.01, 16, 75)

print("Loading data sets...")
train_data, val_data, test_data = get_data(0.01, 0.1, 0)

print("Training CNN...")
cnn_model = CNN_Model()
if torch.cuda.is_available():
    cnn_model.cuda() #USE GPU!

train(cnn_model, train_data, val_data, 0.005, 16)