"""
================================
Recognizing hand-written digits
================================
This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

print(__doc__)

from sklearn import datasets, svm
from skimage import data, color
import matplotlib.pyplot as plt
from skimage.transform import rescale
import numpy as np
from joblib import dump, load
import seaborn as sns
import os
import pandas as pd
import utils
from sklearn import tree
import statistics as st
from numpy.lib.arraysetops import unique
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)

#classification

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

train_data_size = 0.8
test_size = 0.2
collection = []
print("\nTrain:Test:valid\tTrain_Set_Split\t\tBest Gamma\tA_Test_acc\tA_f1_score\n")
#Train-Validation-Test split
X_train, X_test, X_val, y_train,y_test,y_val = utils.create_splits(data,digits.target,train_data_size)

train_splits= [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]       
gammas = [1e-05,0.0001,0.001,0.01]
for i in train_splits:
    if i != 1:
        x_train, Y_train= utils.create_splits_train(X_train,y_train,i)
    else:
        x_train, Y_train = X_train,y_train
    (unique,counts) = np.unique(Y_train,return_counts=True)
    freq = np.asarray((unique,counts)).T
    model_candidates_A = []
    for gmvalue in gammas:
        #Create a classifier: a support vector classifiera
        clf_A = svm.SVC(gamma=gmvalue)

        #Learn the digits on the train data
        clf_A.fit(x_train, Y_train)

        #Predict digits on the validation data
        metrics_val_A = utils.val(clf_A,X_val,y_val,i)

        candidate_A ={"model" : clf_A,"acc_valid" : metrics_val_A['acc'],"f1_valid" : metrics_val_A['f1'], "%_train_data":metrics_val_A["%_train_data"],"gamma" : gmvalue}
        model_candidates_A.append(candidate_A)

        output_folder_A = utils.model_path(test_size,gmvalue,i)
        os.mkdir(output_folder_A)
        dump(clf_A, os.path.join(output_folder_A,"models.joblib"))

    #loading the best model
    #predicting the test data on the best hyperparameter

    max_candidate_A = max(model_candidates_A, key = lambda x :x["f1_valid"])
    best_gamma = max_candidate_A["gamma"]
    
    best_model_folder_A = "models/test_{}_val_{}_hyperparameter_{}_i_{}".format((test_size/2),(test_size/2),best_gamma,i)
    clf_A = load(os.path.join(best_model_folder_A,"models.joblib"))


    #printing accuracies
    metrics_test_A, predicted = utils.test(clf_A,X_test,y_test) 

    print("{} : {} : {}  \t{}\t\t\t{}\t\t{}\t\t{}".format((1-test_size),(test_size/2),(test_size/2),i,best_gamma,round(metrics_test_A['acc'],4),round(metrics_test_A['f1'],4)))

    plt.figure()
    cm = confusion_matrix(y_test,predicted)
    cm
    sns.heatmap(cm,annot=True)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    filename = "Confusion-Matrix for "+str(i*100)+" training data.png"
    plt.savefig(filename)
    if metrics_test_A:
        test_info = {
            "Test Accuracy":metrics_test_A['acc'],
            "Test F1 Score":metrics_test_A['f1']
        }
    max_candidate_A.update(test_info)
    collection.append(max_candidate_A)

df = pd.DataFrame(collection)
plt.figure()
plt.plot(df['%_train_data'],df['Test F1 Score'])
plt.xlabel("Training data %")
plt.ylabel("Test F1 Score")
plt.title("Amount of Training data   Vs.  Test F1 Score")
plt.savefig("dataVSf1.png")
