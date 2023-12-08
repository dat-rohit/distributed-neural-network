import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os 
import argparse
import neptune

from models.model import Network


if __name__ == '__main__':    


    
    run = neptune.init_run(
        project="dat-rohit/PDP-project",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNzJhNzAwMy04OWU5LTQ2NzgtOTA2Zi05YWZiZTNhZTI0MTkifQ==",
    )  # your credentials

    params = {"learning_rate": 0.001, "optimizer": "SGD", "model_name": "nodistmodel", "epochs" : 10}
    run["parameters"] = params

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    print(len(trainloader.dataset))
    print(len(testloader.dataset))

    model = Network()
    model.to(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    for epoch in range(15):  # loop over the dataset multiple times

        running_loss = 0.0
        total_batches = 0
        for i, data in enumerate(trainloader, 0):
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
            total_batches += 1

        average_loss = running_loss / total_batches    
        run["train/loss"].append(average_loss)

        print(f'Epoch {epoch + 1}, Average Training Loss: {average_loss:.3f}')      

    
        # Evaluation loop on the test set
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        test_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        run["val/acc"].append(100 * correct / total)
        run["val/loss"].append(test_loss / len(testloader))

        # Print validation accuracy and loss
        print('Validation Accuracy: %.2f %%' % (100 * correct / total))
        print('Validation Loss: %.3f' % (test_loss / len(testloader)))

run.stop()