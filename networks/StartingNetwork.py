import torch
import torch.nn as nn


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression example. You may need to double check the dimensions :)
    """

    def __init__(self, modelSize):
        print("This is a change!")
        super().__init__()
        self.fc1 = nn.Linear(32, 64) # What could that number mean!?!?!? Ask an officer to find out :)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.lstm = nn.LSTM(modelSize, 32) 
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        x (tensor): the input to the model
        ---Sparsh's test change is here---
        '''
        x = x.float()
        x, y = self.lstm(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x