from sklearn.model_selection import train_test_split
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split, Dataset, TensorDataset

#Split data function

def split_input_arrays(in_data,label_data,size_split):

    xtr, xjunk, ytr, yjunk = train_test_split(in_data,label_data,train_size=size_split)
    xv, xte, yv, yte = train_test_split(xjunk,yjunk, test_size=0.5)
    print('xtrain.shape, ytrain.shape, xval.shape, yval.shape, xtest.shape, ytest.shape')
    print(xtr.shape, ytr.shape, xv.shape, yv.shape, xte.shape, yte.shape)
    return xtr, ytr, xv, yv, xte, yte

#Create dataset and dataloaders from splitted arrays
def get_dataloaders_fromsplitarrays(xtr,ytr,xv,yv,xte,yte,batch_size):

    tr_set = torch.utils.data.TensorDataset(torch.from_numpy(xtr).float(), torch.from_numpy(ytr).float())
    tr_load = torch.utils.data.DataLoader(tr_set, batch_size=batch_size, shuffle=True)

    va_set = torch.utils.data.TensorDataset(torch.from_numpy(xv).float(), torch.from_numpy(yv).float())
    va_load = torch.utils.data.DataLoader(va_set, batch_size=batch_size, shuffle=True)

    te_set = torch.utils.data.TensorDataset(torch.from_numpy(xte).float(), torch.from_numpy(yte).float())
    te_load = torch.utils.data.DataLoader(te_set, batch_size=batch_size, shuffle=True)

    return tr_set, va_set, te_set, tr_load, va_load, te_load 

