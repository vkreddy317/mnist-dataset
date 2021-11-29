#predefined functions
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
import os
from sklearn import tree

def create_splits(data,target,train_data_size):
    
    test_data_size = 0.1
    valid_data_size = 0.1
    train_x,test_x,train_y,test_y = train_test_split(data,target,train_size=train_data_size)
    val_x,test_x,val_y,test_y = train_test_split(test_x,test_y,train_size=(valid_data_size/(valid_data_size + test_data_size)))

    return train_x, test_x, val_x, train_y, test_y, val_y

def create_splits_train(X_train,y_train,i):

   X_train,_,Y_train,_ =  train_test_split(X_train,y_train,train_size=i)
   return X_train,Y_train

def val(clf,X,y,i):

    m = dict()
    predicted = clf.predict(X)
    acc_1 = metrics.accuracy_score(y_pred=predicted, y_true=y)
    f1_score = metrics.f1_score(y_pred=predicted, y_true=y, average='macro')
    m["%_train_data"] = i*100
    m["acc"] = acc_1
    m["f1"] = f1_score

    return m

def test(clf,X,y):

    m = dict()
    predicted = clf.predict(X)
    acc_1 = metrics.accuracy_score(y_pred=predicted, y_true=y)
    f1_score = metrics.f1_score(y_pred=predicted, y_true=y, average='macro')
    m["acc"] = acc_1
    m["f1"] = f1_score

    return m, predicted

def model_path(test_size,value,i):
    
    output_folder = "models/test_{}_val_{}_hyperparameter_{}_i_{}".format((test_size/2),(test_size/2),value,i)
    return output_folder
