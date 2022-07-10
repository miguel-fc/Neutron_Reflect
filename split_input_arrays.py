#Split data function

from sklearn.model_selection import train_test_split

def split_input_arrays(in_data,label_data,size_split):

    xtr, xjunk, ytr, yjunk = train_test_split(in_data,label_data,train_size=size_split)
    xv, xte, yv, yte = train_test_split(xjunk,yjunk, test_size=0.5)
    print('xtrain.shape, ytrain.shape, xval.shape, yval.shape, xtest.shape, ytest.shape')
    print(xtr.shape, ytr.shape, xv.shape, yv.shape, xte.shape, yte.shape)
    return xtr, ytr, xv, yv, xte, yte