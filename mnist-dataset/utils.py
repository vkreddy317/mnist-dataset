import sklearn
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,svm,metrics
from sklearn.model_selection import train_test_split

def preprocessing(image,rescale_factor):
    resized_images = []
    for img in image:
        resized_images.append(transform.rescale(img,rescale_factor,anti_aliasing=False))
    return resized_images

def create_split(data,target,train_size,valid_size,test_size):
    train_X,test_X,train_Y,test_Y = train_test_split(data,target,test_size=test_size + valid_size,shuffle=False)
    val_X,test_X,val_Y,test_Y = train_test_split(test_X,test_Y,test_size =(10/3)*test_size,shuffle = False)

    return train_X,train_Y,test_X,test_Y,val_X,val_Y