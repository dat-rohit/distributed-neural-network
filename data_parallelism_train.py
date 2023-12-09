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

import neptune
import time

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

criterion = nn.CrossEntropyLoss()


# Define global variables to track time
data_loading_time = 0.0
training_time = 0.0
evaluation_time = 0.0
mpi_communication_time_parent = 0.0
mpi_communication_time_children = 0.0



def partition_dataset(dataset, rank, size):
    total_size = len(dataset)
    partition_size = total_size // (size-1)
    indices = list(range((rank-1) * partition_size, (rank) * partition_size))
    return data.Subset(dataset, indices)


def main(args):

    

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    model = Network()

    if rank != 0:
        # Only the child workers get a portion of the dataset
        start_time = time.time()
        train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
        train_subset = partition_dataset(train_dataset, rank, size)

        trainLoader = data.DataLoader(
            train_subset,
            shuffle=True,
            num_workers=1,
            batch_size=int(args.bs)
        )
        end_time = time.time()
    
        

        print("(Loaded Train Dataset for worker {0} of length {1})".format(rank, len(trainLoader.dataset)))
    else:
        trainLoader = None
        start_time = time.time()
        testLoader = data.DataLoader(
            torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform),
            batch_size=int(args.bs)
        )
        end_time = time.time()
        

    global data_loading_time
    data_loading_time += end_time - start_time
    global mpi_communication_time_parent
    global mpi_communication_time_children

  
    if rank == 0:

        log_filename = f"bs{args.bs}_log_epochs{args.epochs}_proc{args.nb_proc}_parent.txt"
        log_path = os.path.join("log", log_filename)    

        run = neptune.init_run(
            project="dat-rohit/PDP-project",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNzJhNzAwMy04OWU5LTQ2NzgtOTA2Zi05YWZiZTNhZTI0MTkifQ==",
        )  # your credentials

        params = {"learning_rate": 0.001, "optimizer": "SGD", "model_name": "distmodel", "epochs" : int(args.epochs)}
        run["parameters"] = params
        
        for i in range(int(args.epochs)):
            print("Starting epoch ", i)
            start_time=time.time()
            [comm.send(model.state_dict(), k) for k in range(1, size)]
            end_time=time.time()
            mpi_communication_time_parent+=end_time-start_time
            run_parent(model=model, comm=comm, size=size, testLoader=testLoader, run=run)
    
        print("Eval data loading time: {0}".format(data_loading_time))
        print("Time spent on evaluation: {0}".format(evaluation_time))
        print("Time spent on parent communication and param sync: {0}".format(mpi_communication_time_parent))
        with open(log_path, "w") as log_file:
            log_file.write("Eval data loading time: {0}\n".format(data_loading_time))
            log_file.write("Time spent on evaluation: {0}\n".format(evaluation_time))
            log_file.write("Time spent on parent communication and param sync: {0}\n".format(mpi_communication_time_parent))


    else:

        for i in range(int(args.epochs)):
            data_model=comm.recv()
            start_time=time.time()
            model.load_state_dict(data_model)
            end_time=time.time()
            mpi_communication_time_children+=end_time-start_time

            run_child(model=model, comm=comm, trainLoader=trainLoader)
        if rank==2:
            log_filename = f"bs{args.bs}_log_epochs{args.epochs}_proc{args.nb_proc}_children.txt"
            log_path = os.path.join("log", log_filename)    
            print("Training data loading time: {0}".format(data_loading_time))
            print("Time spent on training: {0}".format(training_time))
            print("Time spent on children communication: {0}".format(mpi_communication_time_children))
            with open(log_path, "w") as log_file:
                log_file.write("Train data loading time: {0}\n".format(data_loading_time))
                log_file.write("Time spent on training: {0}\n".format(training_time))
                log_file.write("Time spent on children communication: {0}\n".format(mpi_communication_time_children))




def eval(model, testLoader, run):
    start_time = time.time()
    global evaluation_time

    model.eval()
    with torch.no_grad():
        losses=[]
        corrects = 0
        total_samples = 0
        for i, (input, target) in enumerate(testLoader, 0):
            output = model(input)
            loss = criterion(output, target)
            losses.append(loss.item())


            # calculate accuracy
            corrects += (torch.argmax(output, dim=1) == target).sum().item()
            total_samples += target.size(0)

        avg_loss = np.mean(losses)
        accuracy_val = 100 * corrects / total_samples
        end_time = time.time()
        evaluation_time += end_time - start_time
        run["val/loss"].append(avg_loss)
        run["val/acc"].append(accuracy_val)

        print("Validation loss of updated master model: ", np.mean(losses))

def run_child(model, comm, trainLoader):

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    total_loss = 0.0
    total_batches = 0
    start_time = time.time()


    for i, (input, target) in enumerate(trainLoader, 0):

        optimizer.zero_grad()
        output=model(input)
        loss=criterion(output, target)
        loss.backward()
        optimizer.step()

        # accumulate the loss
        total_loss += loss.item()
        total_batches += 1
    
    end_time=time.time()
    global training_time
    training_time+=end_time-start_time   

    start_time=time.time()
    comm.send({'state_dict': model.state_dict(), 'total_loss': total_loss}, 0)
    end_time=time.time()
    global mpi_communication_time_children
    mpi_communication_time_children+=end_time-start_time
#TODO code run_parent anb main function
#Code gradient upload and parameter synchronization
#Code dataset partitioner


def run_parent(model, comm, size, testLoader, run):
    state_dicts = []    
    total_batches = 0
    total_losses = 0.0
    global mpi_communication_time_parent

    #receive the model parameters of all children process
    for p in range(size-1):
        data = comm.recv()
        start_time=time.time()
        state_dicts.append(data['state_dict'])
        end_time=time.time()
        mpi_communication_time_parent+=end_time-start_time
        total_losses += data['total_loss']
        total_batches += len(data['state_dict'])  
        print("(Received a trained model from process {0} of {1} workers...)".format(p+1, size-1))

    start_time=time.time()
    #perform Parameter Synchronization
    avg_state_dict = OrderedDict()
    for key in state_dicts[0].keys():
        avg_state_dict[key] = sum([sd[key] for sd in state_dicts]) / float(size-1)

    #update the master model with the synchronized parameters
    print("* Averaging models...")
    model.load_state_dict(avg_state_dict)
    end_time=time.time()
    mpi_communication_time_parent+=end_time-start_time

    global_avg_loss = total_losses / float(total_batches)
    print(f"Global Average Training Loss: {global_avg_loss}")
    run["train/loss"].append(global_avg_loss)

    
    print("evaluating model")
    eval(model, testLoader, run)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", dest="lr", default=0.001)
    parser.add_argument("--momentum", dest="momentum", default=0.9)
    parser.add_argument("--batch-size", dest="bs", default=16)
    parser.add_argument("--epochs", dest="epochs", default=25)
    parser.add_argument("--nb-proc", dest="nb_proc", default=4)

    args = parser.parse_args()
    main(args)
