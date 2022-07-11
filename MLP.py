# Import Python related required packages
import io
import cv2
import gdown
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.express as px
from sklearn.metrics import mean_squared_error
from scipy.stats import gaussian_kde, norm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
import pandas as pd
from tqdm import tqdm
import pickle

#Import torch related packages
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split, Dataset, TensorDataset
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


#Defining a Multilayer Perceptron, MLP.
class MLP(nn.Module):

  def __init__(self,dim):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(d, 500),
      nn.ReLU(),
      nn.Linear(500, nl),
      # nn.ReLU(),
      # nn.Linear(32, nl),
    )

  def forward(self, x):
    return self.layers(x)

### Training function
def fit(model, device, dataloader, loss_fn, optimizer):
    model.train()
    train_loss = []
    for data,label in dataloader: 
        img = data
        img = img.to(device)
        # print(img.shape)
        label = label.to(device)
        out_label = model(img)
        loss = loss_fn(out_label, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

    

### Valid function
def val(model, device, dataloader, loss_fn):
    model.eval()
    with torch.no_grad(): 
        list_decoded_img = []
        list_img = []
        for  data, label in dataloader:
            img = data
            img = img.to(device)
            label = label.to(device)
            out_label = model(img)
            list_decoded_img.append(out_label.cpu())
            list_img.append(label.cpu())
        list_decoded_img = torch.cat(list_decoded_img)
        list_img = torch.cat(list_img) 
        val_loss = loss_fn(list_decoded_img, list_img)
    return val_loss.data

