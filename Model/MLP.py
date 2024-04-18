import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(42)

# Four layer neural network with RELU between hidden layers and softmax on output
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(31, 62) 
        self.fc2 = nn.Linear(62,62)
        self.fc3 = nn.Linear(62,31)
        self.fc4 = nn.Linear(31, 12)
        self.fc5 = nn.Linear(12, 2) 

    def forward(self, x):
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = F.softmax(self.fc5(out))

        return out 
    
# TODO: 
# - Create dataloader and get train/test splits