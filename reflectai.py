# Import Python related required packages
import io
import os
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

### Training function
def fit(encoder, decoder, device, dataloader, loss_fn, optimizer):
    encoder.train().to(device)
    decoder.train().to(device)
    train_loss = []
    for data,label in dataloader: 
        img = data
        img = img.view(img.size(0), -1).to(device)  
        label = label.to(device)
        latent = encoder(img)
        decoded_img = decoder(latent)
        loss = loss_fn(decoded_img, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

    ### Valid function
def val(encoder, decoder, device, dataloader, loss_fn):
    encoder.eval().to(device)
    decoder.eval().to(device)
    with torch.no_grad(): 
        list_decoded_img = []
        list_img = []
        for  data, label in dataloader:
            img = data
            img = img.view(img.size(0), -1).to(device) 
            label = label.to(device)
            latent = encoder(img)
            decoded_img = decoder(latent)
            list_decoded_img.append(decoded_img.cpu())
            list_img.append(img.cpu())
        list_decoded_img = torch.cat(list_decoded_img)
        list_img = torch.cat(list_img) 
        val_loss = loss_fn(list_decoded_img, list_img)
    return val_loss.data


#test and plot outputs
def test(encoder,decoder,dataset,device,n=10,loss_fn):
    plt.figure(figsize=(26,5.5))
    for i in range(10):
      ax = plt.subplot(2,n,i+1)
      img,_ = dataset[i]
      #Notice that below i'm loading an image only, so it needs to be flatten
      #before entering the network
      img = torch.flatten(img).to(device)
      encoder.eval().to(device)
      decoder.eval().to(device)
      with torch.no_grad():
         decoded_img  = decoder(encoder(img))
         loss = loss_fn(decoded_img,img)
         print('For image {}, the loss = {}'.format(i,loss.data))
      plt.plot(img.cpu().reshape(in_d1,in_d2).numpy()[0],img.cpu().reshape(in_d1,in_d2).numpy()[1]) 
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n) 
      plt.plot(decoded_img.cpu().reshape(in_d1,in_d2).numpy()[0],decoded_img.cpu().reshape(in_d1,in_d2).numpy()[1]) 
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.show()  


def plot_ae_outputs(encoder,decoder,dataset,device,n=10):
    plt.figure(figsize=(26,5.5))
    for i in range(10):
      ax = plt.subplot(2,n,i+1)
      img,_ = dataset[i]
      #Notice that below i'm loading an image only, so it needs to be flatten
      #before entering the network
      img = torch.flatten(img).to(device)
      encoder.eval().to(device)
      decoder.eval().to(device)
      with torch.no_grad():
        decoded_img = decoder(encoder(img))
      plt.plot(img.cpu().reshape(2,128).numpy()[0],img.cpu().reshape(2,128).numpy()[1]) 
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n) 
      plt.plot(decoded_img.cpu().reshape(2,128).numpy()[0],decoded_img.cpu().reshape(2,128).numpy()[1]) 
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.show()  