# Enhancing CNN Training on CIFAR-10 Through MPI Parallelization


This repository contains the source code for training a Convolutional Neural Network (CNN) using different parallelization strategies. Below is a brief overview of the key components:

## Models

### [models/model.py](models/model.py)

This file contains the implementation of the CNN architecture used in the training process.

## Training Scripts

### [single_proc_train.py](single_proc_train.py)

This script implements the training of the model without any parallelization. It serves as a baseline for performance comparison with parallelized approaches.

### [model_replication_train.py](model_replication_train.py)

In this script, the training is performed with only model replication over processes, without data parallelism. It's designed to showcase the impact of replicating the model across multiple processes.

### [data_parallelism_train.py](data_parallelism_train.py)

The core script that implements the data parallelism approach. It includes time measurement functionalities and a fault tolerance simulation. This approach distributes the computational workload across multiple processes, aiming to improve training efficiency.

## Usage
```python
mpiexec -n {number of process} python data_parallelism_train.py --nb-proc {number of process} 
```
