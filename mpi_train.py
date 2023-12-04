# File: train_mpi.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.distributed as dist
import torchvision.transforms as transforms
import numpy as np
import os 
import argparse
import torchvision


from models.model import Network
from mpi4py import MPI

def train(model, dataloader, criterion, optimizer, device):
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

def evaluate(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct, total, test_loss


def average_gradients(model):
    #size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Process {rank}/{size} is using device {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 4

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=size, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    model = Network()
    model.to(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(6):
        #train(model, train_loader, criterion, optimizer, device)

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Synchronize before gradient sharing
            comm.Barrier()

            average_gradients(model)
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
       

        # Evaluation on the test set
        correct, total, test_loss = evaluate(model, test_loader, criterion, device)
        print(correct, total, test_loss)
        print('scores')
        # Reduce accuracy and loss across all processes
        total_correct = comm.allreduce(correct, op=MPI.SUM)
        total_samples = comm.allreduce(total, op=MPI.SUM)
        total_loss = comm.allreduce(test_loss, op=MPI.SUM)

        if rank == 0:
            # Print validation accuracy and loss
            print(f'Epoch {epoch + 1}:')
            print('Validation Accuracy: %.2f %%' % (100 * total_correct / total_samples))
            print('Validation Loss: %.3f' % (total_loss / len(test_loader)))
