import numpy as np
from numpy.lib.arraysetops import unique
import pandas as pd

import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score

from joblib import load,dump
import os

digits = load_digits()

n_sample = len(digits.images)
data = digits.images.reshape((n_sample,-1))

def create_train_test_split(X,Y,train_size = 0.8):
    test_size = 1-train_size
    assert train_size + test_size == 1
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=train_size)
    return X_train,X_test,Y_train,Y_test

def get_SVM_metrics(train_X,train_Y,test_X,test_Y,hyperparameter):
    clf = SVC(gamma=hyperparameter)
    clf.fit(train_X,train_Y)
    predict = clf.predict(test_X)
    predict_train = clf.predict(train_X)
    acc_train = accuracy_score(y_true=train_Y,y_pred=predict_train)
    acc = accuracy_score(y_true=test_Y,y_pred=predict)
    f1 = f1_score(y_true=test_Y,y_pred=predict,average='macro')
    return {'acc':acc,'f1':f1},clf,acc_train

def get_test_metrics(model,test_x,test_y):
    predict = model.predict(test_x)
    acc = accuracy_score(y_true=test_y,y_pred=predict)
    f1 = f1_score(y_true=test_y,y_pred=predict,average='macro')
    return {'acc':acc,'f1':f1},predict


# 70:15:15 for train:validation:test Split
train_data_size = 0.7
test_data_size = 0.15
valid_data_size = 0.15

hyperparameters = [0.00001,0.0001,0.001,0.01,0.1,1,2,5,10,15,50]
best_model_collection = []
overfit_data_collection = []
for i in range(3):
    train_x,test_x,train_y,test_y = train_test_split(data,digits.target,train_size=train_data_size)
    val_x,test_x,val_y,test_y = train_test_split(test_x,test_y,train_size=(valid_data_size/(valid_data_size + test_data_size)))
    model_collection = []
    overfit = []
    for gamma_val in hyperparameters:
        val_score,model,train_acc = get_SVM_metrics(train_X=train_x,train_Y=train_y,test_X=val_x,test_Y=val_y,hyperparameter=gamma_val)
        if val_score:
             candidate = {
                "Round No.":i,
                "Validation Acc.":val_score['acc'],
                "Validation F1":val_score['f1'],
                "Hyperparameter":gamma_val
            }
        model_collection.append(candidate)
        try:    # Preventing form the error if model is already present 
                output_folder = "./models/round_no_{}_svm_gamma_{}".format(i,gamma_val)
                os.mkdir(output_folder)
                dump(model,os.path.join(output_folder,'model.joblib'))
        except:
                pass
        
        test_score,_ = get_test_metrics(model=model,test_x=test_x,test_y=test_y)
        print("For Round ",i," Gamma :- ",gamma_val)
        acc_dif = (train_acc-test_score['acc'])
        overfit_check = {
            "Accuracy Diff.":acc_dif,
            "Round":i,
            "Hyperparameter":gamma_val,
            "Train Accuracy":train_acc,
            "Test Accuracy":test_score['acc'],
            "Validation Accuracy":val_score['acc']
        }
        overfit.append(overfit_check)
    max_valid_f1_score = max(model_collection,key=lambda x:x['Validation F1'])
        #," Acc diff :- ",(train_acc-test_score['acc'])
    worst_overfitting_models = max(overfit,key=lambda x:x['Accuracy Diff.'])
    best_model_collection.append(max_valid_f1_score)
    overfit_data_collection.append(worst_overfitting_models)


df1 = pd.DataFrame(best_model_collection)
df2 = pd.DataFrame(overfit_data_collection)
df1.to_csv("Best_hyperparameter.csv")
df2.to_csv("Overfitting_hyperparameter.csv")
print("\n")
print("++++++++++++++ Best Hyperparameter ++++++++++++++++++++++")
print(df1)

print("\n")
print("+++++++++++++++++ Overfitting +++++++++++++++++++++++++++++")
print(df2)
print("\n")