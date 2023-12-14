import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import flwr as fl
from collections import OrderedDict


def train(net, trainloader, epochs: int, DEVICE: str ='cuda'):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader, DEVICE: str = 'cuda'):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

import os

def read_data(idx, is_train=True, ROOT="./tiny_camelyon17"):
    if is_train:
        train_data_dir = os.path.join(ROOT, "train/")

        train_file = train_data_dir + str(idx) + ".npz"
        with open(train_file, "rb") as f:
            train_data = np.load(f, allow_pickle=True)["data"].tolist()

        return train_data

    else:
        test_data_dir = os.path.join(ROOT, "test/")

        test_file = test_data_dir + str(idx) + ".npz"
        with open(test_file, "rb") as f:
            test_data = np.load(f, allow_pickle=True)["data"].tolist()

        return test_data
    
def load_datasets(num_clients: int, val=False):
    # describe this function
    """ 
    this function loads the data from the path and splits it into
    train, validation and test sets.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),]
    )
    
    trainloaders, valloaders, testloaders = [], [], []
    for i in range(num_clients):
        trainset = read_data(i, True)
        xtrain, ytrain = np.array(trainset['x']), trainset['y']
        if val:
            xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.25, random_state=42, stratify=ytrain)
        testset = read_data(i, False)
        xtest, ytest = np.array(testset['x']), testset['y']
        train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(xtrain), torch.from_numpy(ytrain))
        if val:
            val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(xval), torch.from_numpy(yval))
        test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(xtest), torch.from_numpy(ytest))
        trainloaders.append(DataLoader(train_dataset, batch_size=16, shuffle=True))
        if val:
            valloaders.append(DataLoader(val_dataset, batch_size=16))
        testloaders.append(DataLoader(test_dataset, batch_size=16))

    return trainloaders, valloaders, testloaders

def get_weights(model: torch.nn.ModuleList) -> fl.common.NDArrays:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model: torch.nn.ModuleList, weights: fl.common.NDArrays) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)

def create_global_testset():
    global_xtest, global_ytest = [], []
    for client in range(5):
        data = read_data(client, False)
        xtest, ytest = np.array(data['x']), data['y']
        # take random 20% of the data
        #_, xtest, _, ytest = train_test_split(xtest, ytest, test_size=0.2, random_state=42, stratify=ytest)
        global_xtest.append(xtest)
        global_ytest.append(ytest)
        
    global_xtest = np.concatenate(global_xtest)
    global_ytest = np.concatenate(global_ytest) 
    global_test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(global_xtest), torch.from_numpy(global_ytest))
    global_testloader = DataLoader(global_test_dataset, batch_size=16)
    return global_testloader