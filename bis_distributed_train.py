import torch
import torch.nn as nn
import torch.utils.data as data
import torch.distributed as dist
from torch.multiprocessing import Process
import torchvision
import torchvision.transforms as transforms

# MPI library:
from mpi4py import MPI

# convnet model that classifies MNIST:
from models.model import Network

# other utilities:
import numpy as np
import os
import argparse
from collections import OrderedDict


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

criterion = nn.CrossEntropyLoss()


def eval(model):

    model.eval()
    with torch.no_grad():
        testLoader = data.DataLoader(
        torchvision.datasets.CIFAR10(root='./data', train=False, download=True, batch_size=args.bs, transform=transform))
        losses=[]
        for (input, target) in enumerate(testLoader):
            loss=criterion(model(input), target)
            losses.append(loss.item())

        print("Validation loss of updated master model: ", np.mean(losses))

def run_child(model, comm):

    trainLoader = data.DataLoader(
        torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform),
        shuffle=True,
        num_workers=1,
        batch_size=args.bs
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    for i, (input, target) in enumerate(trainLoader):

        optimizer.zero_grad()
        output=model(input)
        loss=criterion(output, target)
        loss.backward()
        optimizer.step()

    comm.send(model.state_dict(), 0)

#TODO code run_parent anb main function
#Code gradient upload and parameter synchronization
#Code dataset partitioner


def run_parent(model, comm, size):
    state_dicts = []


    #receive the model parameters of all childrem process
    for p in range(size-1):
        state_dicts.append(comm.recv())

    #perform Parameter Synchronization
    avg_state_dict = OrderedDict()
    for key in state_dicts[0].keys():
        avg_state_dict[key] = sum([sd[key] for sd in state_dicts]) / float(size-1)
    
    #update the master model with the synchronized parameters
    model.load_state_dict(avg_state_dict)
    
    eval(model)

def main(args):


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    model = Network()

    if rank == 0:

        for i in range(args.epochs):
            print("Starting epoch ", i)
            [comm.send(model.state_dict(), k) for k in range(1, size)]
            run_parent(model=model, comm=comm, size=size)
    else:
        for i in range(args.epochs):
            model.load_state_dict(comm.recv())
            run_child(model=model, comm=comm)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", dest="lr", default=0.001)
    parser.add_argument("--momentum", dest="momentum", default=0.9)
    parser.add_argument("--batch-size", dest="bs", default=16)
    parser.add_argument("--epochs", dest="epochs", default=10)
    args = parser.parse_args()
