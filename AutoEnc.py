#Import torch related packages
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split, Dataset, TensorDataset
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


class Encoder(nn.Module):
    
    def __init__(self,encoded_space_dim,dim1,dim2,num_layers):
        super().__init__()
        
        self.layers = []
        self.layers.append(nn.Linear(dim1 * dim2, 100))
        self.layers.append(nn.ReLU(True))
        for i in range(num_layers):
          self.layers.append(nn.Linear(100,100))
          self.layers.append(nn.ReLU(True))
        
        self.layers.append(nn.Linear(100,encoded_space_dim))

        self.encoder = nn.Sequential(*self.layers)
        
    def forward(self, x):
        x = self.encoder(x)
        return x

class Decoder(nn.Module):
    
    def __init__(self,encoded_space_dim,dim1,dim2,num_layers):
        super().__init__()

        self.layers = []
        # self.decoder = nn.Sequential(
        self.layers.append(nn.Linear(encoded_space_dim, 100))
        self.layers.append(nn.ReLU(True))
        for i in range(num_layers):
          self.layers.append(nn.Linear(100, 100))
          self.layers.append(nn.ReLU(True))
        
        self.layers.append(nn.Linear(100, dim1 * dim2))
        self.decoder = nn.Sequential(*self.layers)
        
    def forward(self, x):
        x = self.decoder(x)
        return x