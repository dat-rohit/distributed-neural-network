import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim




class Network(nn.Module):

    def __init__(self):
        super().__init__()

        self.convolutions = nn.Sequential(
             nn.Conv2d(3, 6, 5),
             nn.MaxPool2d(2, 2),
             nn.Conv2d(6, 16, 5),
             nn.MaxPool2d(2, 2))
        
        self.fc = nn.Sequential(
            nn.Linear(6*16*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10))
        
    def forward(self, x):

        x = self.convolutions(x)
        x = self.fc(x)

        return x
    


    

