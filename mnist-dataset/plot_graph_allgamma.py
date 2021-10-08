import sklearn
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,svm,metrics
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

n_sample = len(digits.images)
data = digits.images.reshape((n_sample,-1))

resize_images_size = [0.25,0.5,1,2,4,8]
hyperparameter_values =[0.00001,0.0001,0.001,0.05,1,2,5,10]
print("\t\tResize \t\tDataset \t\tgamma_value \t\tAccuracy")
for i in range(len(resize_images_size)):
    resized_images = []
    testacc = []
    valacc=[]
    for img in digits.images:
        resized_images.append(transform.rescale(img,resize_images_size[i],anti_aliasing=False))
    for j in range(len(hyperparameter_values)):
        resized_images = np.array(resized_images)
        data = resized_images.reshape((n_sample,-1))
        model = svm.SVC(gamma = hyperparameter_values[j])
        train_X,test_X,train_Y,test_Y = train_test_split(data,digits.target,test_size=0.3,shuffle=False)
        val_X,test_X,val_Y,test_Y = train_test_split(test_X,test_Y,test_size =0.5,shuffle = False)
        model.fit(train_X,train_Y)
        predict_val = model.predict(val_X)
        acc_val = metrics.accuracy_score(y_pred= predict_val,y_true = val_Y)
        valacc.append(acc_val)
        predict_test = model.predict(test_X)
        acc_test = metrics.accuracy_score(y_pred= predict_test,y_true = test_Y)
        testacc.append(acc_test)
        print("\t\t ",resize_images_size[i] ,"\t\tValidation Set","\t\t",hyperparameter_values[j],"\t\t",acc_val*100)
        print("\t\t ",resize_images_size[i] ,"\t\tTest Set","\t\t",hyperparameter_values[j],"\t\t",acc_test*100)
