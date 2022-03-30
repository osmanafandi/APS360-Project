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


def get_data_old(len_train_data, len_val_data, len_test_data):
    # ***** Specify the path to final dataset folder on your loca machine ******
    data_path = "./Final Dataset"
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])

    # Seperate for training, validation, and test data
    dataset = torchvision.datasets.ImageFolder(data_path, transform=transform)
    num_train = math.floor(len(dataset) * len_train_data)
    num_val = math.floor(len(dataset) * len_val_data)
    num_test = math.floor(len(dataset) * len_test_data)
    print("Number of images for training: ", num_train)
    print("Number of images for validation: ", num_val)
    print("Number of images for test: ", num_test)
    dummy_len = len(dataset) - (num_test + num_train + num_val)

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

    # Split into train and validation
    train_set, val_set, test_set, dummy = torch.utils.data.random_split(dataset, [num_train, num_val, num_test,
                                                                                  dummy_len])  # 80%, 10%, 10% split

    # return train_loader, val_loader, test_loader
    return train_set, val_set, test_set


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


class ANNBaseline(nn.Module):
    def __init__(self):
        super(ANNBaseline, self).__init__()
        self.name = "Baseline"
        self.layer1 = nn.Linear(3 * 224 * 224, 84)  # images are rgb 224 x 224 pixels
        self.layer2 = nn.Linear(84, 28)
        self.layer3 = nn.Linear(28, 5)  # 7 outputs

    def forward(self, img):
        flattened = img.view(-1, 3 * 224 * 224)
        activation1 = self.layer1(flattened)
        activation1 = F.relu(activation1)
        activation2 = self.layer2(activation1)
        activation2 = F.relu(activation2)
        output = self.layer3(activation2)
        output = output.squeeze(1)
        return output


class CNNBaseline(nn.Module):
    def __init__(self):
        super(CNNBaseline, self).__init__()
        self.conv1 = nn.Conv2d(3, 5,
                               5)  # in_channels, out_channels, kernel_size, ((224-5+1)/2) = 110, use to connect layer!
        self.conv2 = nn.Conv2d(5, 10, 3)  # in_channels, out_channels, kernel_size, ((110-3+1)/2) use to connect layer.
        self.conv3 = nn.Conv2d(10, 25, 3)
        self.pool = nn.MaxPool2d(2, 2)  # kernel_size, stride
        self.fc1 = nn.Linear(25 * 26 * 26, 32)  # 10 * ((110-5+1)/2) * ((110-5+1)/2)
        self.fc2 = nn.Linear(32, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 25 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze(1)  # Flatten to the batch size
        return x


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

        output = model(imgs)

        # select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total


def train_baseline(model, train_data, val_data, learning_rate=0.001, batch_size=64, num_epochs=1):
    torch.manual_seed(1000)  # set the random seed
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
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
        train_acc.append(get_accuracy(model, train_data, batch_size))  # compute training accuracy
        val_acc.append(get_accuracy(model, val_data, batch_size))  # compute validation accuracy
        print(("Epoch {}: Train acc: {} |" + "Validation acc: {}").format(
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
    plt.plot(range(1, num_epochs + 1), train_acc, label="Train")
    plt.plot(range(1, num_epochs + 1), val_acc, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))


# baseline_model = CNNBaseline()
# if torch.cuda.is_available():
#     baseline_model.cuda() #USE GPU!

# print("Testing for overfit (sanity check)...")
# train_data, val_data, test_data = get_data(0.001, 0.001, 0.001) #load small dataset for overfit test
# train_baseline(baseline_model, train_data, val_data, 0.01, 64, 150)

print("Loading data sets...")
train_data, val_data, test_data = get_data_old(0.3, 0.03, 0)
# train_data = get_data("./DatasetAugmented", 0.8)
# val_data = get_data("./Validation Images", 1.0)

print("Training baseline...")
baseline_model = CNNBaseline()
if torch.cuda.is_available():
    baseline_model.cuda()  # USE GPU!

train_baseline(baseline_model, train_data, val_data, 0.001, 64, 15)
