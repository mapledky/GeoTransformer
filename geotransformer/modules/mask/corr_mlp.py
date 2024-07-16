import torch
import torch.nn as nn
import torch.nn.functional as F

class CorrMlp(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CorrMlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        #x: (n, m)
        x = x.view(-1, 1)#(n*m, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = x.mean()
        return x