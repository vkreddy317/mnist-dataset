import sklearn
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,svm,metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import statistics
import os
from joblib import dump
import pandas as pd

digits = datasets.load_digits()

n_sample = len(digits.images)
data = digits.images.reshape((n_sample,-1))

gamma_values =[0.00001,0.0001,0.001,0.05,1,2,5,10]
max_depth_values_tree = [5,10,15,20,25,30,50,100]

models_svm = []
models_tree = []

acc_best_SVM = []
f1_best_SVM = []
acc_best_tree = []
f1_best_tree = []

performance_data = []


def create_split(X,y,train_size=0.7,test_size=0.20,val_size=0.10):
    X_train,X_test,Y_train,Y_test = train_test_split(X,y,train_size=train_size)
    X_val,X_test,Y_val,Y_test = train_test_split(X_test,Y_test,test_size=(test_size/(test_size+val_size)))
    return X_train,Y_train,X_val,Y_val,X_test,Y_test

def get_val_metrics(model,train_X,train_Y,val_X,val_Y):
    model.fit(train_X,train_Y)
    pred = model.predict(val_X)
    val_f1 = metrics.f1_score(y_pred=pred,y_true=val_Y,average='macro')
    val_acc = metrics.accuracy_score(y_pred=pred,y_true=val_Y)
    return {'f1_valid':val_f1,'acc_valid':val_acc}
for round in range(5):
    for gamma_val,max_depth_val in zip(gamma_values,max_depth_values_tree):
        X_train,Y_train,X_val,Y_val,X_test,Y_test = create_split(X=data,y=digits.target)
        model = svm.SVC(gamma=gamma_val)
        acc_svm = get_val_metrics(model=model,train_X=X_train,train_Y=Y_train,val_X=X_val,val_Y=Y_val)
        #print("Round ",round,"For SVM ",gamma_val," ",acc_svm)
        if acc_svm:
            candidate_svm = {
                "Round":round,
                "acc_valid":acc_svm['acc_valid'],
                "f1_valid":acc_svm['f1_valid'],
                "hyperparameter":gamma_val
            }
        models_svm.append(candidate_svm)

        tree = DecisionTreeClassifier(max_depth=max_depth_val)
        acc_tree = get_val_metrics(model=tree,train_X=X_train,train_Y=Y_train,val_X=X_val,val_Y=Y_val)
        #print("Round ",round,"For Tree ",max_depth_val," ",acc_tree)
        if acc_tree:
            candidate_tree = {
                "Round":round,
                "acc_valid":acc_tree['acc_valid'],
                "f1_valid":acc_tree['f1_valid'],
                "hyperparameter":max_depth_val
            }
        models_tree.append(candidate_tree)

    max_valid_f1_model_candidate_SVM = max(
                        models_svm, key=lambda x: x["f1_valid"]
                    )
    max_valid_f1_model_candidate_tree = max(
                        models_tree, key=lambda x: x["f1_valid"]
                    )
    acc_best_tree.append(max_valid_f1_model_candidate_tree['acc_valid'])
    f1_best_tree.append(max_valid_f1_model_candidate_tree['f1_valid'])
    acc_best_SVM.append(max_valid_f1_model_candidate_SVM['acc_valid'])
    f1_best_SVM.append(max_valid_f1_model_candidate_SVM['f1_valid'])
    performance = {
        'Round No.':round,
        'SVM gamma':max_valid_f1_model_candidate_SVM['hyperparameter'],
        'SVM Accuracy':max_valid_f1_model_candidate_SVM['acc_valid'],
        'SVM f1_score':max_valid_f1_model_candidate_SVM['f1_valid'],
        'Tree max_depth':max_valid_f1_model_candidate_tree['hyperparameter'],
        'Tree Accuracy':max_valid_f1_model_candidate_tree['acc_valid'],
        'Tree f1_score':max_valid_f1_model_candidate_tree['f1_valid']
    }
    performance_data.append(performance)
    try:
        # Dumping my best current model            
        output_folder = "./models/tt_{}_val_{}_round_{}_svm_hyper_{}".format(0.2,0.1,round,max_valid_f1_model_candidate_SVM['hyperparameter'])
        os.mkdir(output_folder)
        dump(model,os.path.join(output_folder,'model.joblib'))

        output_folder = "./models/tt_{}_val_{}_round_{}_tree_hyper_{}".format(0.2,0.1,round,max_valid_f1_model_candidate_tree['hyperparameter'])
        os.mkdir(output_folder)
        dump(model,os.path.join(output_folder,'model.joblib'))
    except:
        pass

df = pd.DataFrame(performance_data)
print(df)
print("\n \n")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("++++++++++++++++++++++++++++++++++++++  Accuracy  +++++++++++++++++++++++++++++++++++++++++++++")
print("Mean Accuracy of SVM over 5 Rounds :- ",statistics.mean(acc_best_SVM))
print("Mean Accuracy of Tree over 5 Rounds :- ",statistics.mean(acc_best_tree))
print("Standard Deviation of Accuracy of SVM over 5 Rounds :- ",statistics.pstdev(acc_best_SVM))
print("Standard Deviation of Accuracy of Tree over 5 Rounds :- ",statistics.pstdev(acc_best_tree))
print("\n \n")
print("++++++++++++++++++++++++++++++++++++++  f1-Score  +++++++++++++++++++++++++++++++++++++++++++++")
print("Mean f1-score of SVM over 5 Rounds :- ",statistics.mean(f1_best_SVM))
print("Mean f1-score of Tree over 5 Rounds :- ",statistics.mean(f1_best_tree))
print("Standard Deviation of f1-score of SVM over 5 Rounds :- ",statistics.pstdev(f1_best_SVM))
print("Standard Deviation of f1-score of Tree over 5 Rounds :- ",statistics.pstdev(f1_best_tree))