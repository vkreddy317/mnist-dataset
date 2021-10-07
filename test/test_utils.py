from sklearn import datasets
import random
import numpy as np
digits = datasets.load_digits()
from utils import Experiment_model


def test_overfitting_data():
    images = digits.images
    random_subSample = random.sample(range(0,1796),20)
    X = []
    Y = []
    for i in random_subSample:
        X.append(digits.images[i])
        Y.append(digits.target[i])
    X = np.array(X)
    Y = np.array(Y)
    X = X/255
    X = X.reshape((20, -1))
    metrics_model = Experiment_model(train_X = X,train_Y = Y,val_X = X, val_Y = Y)
    assert max(metrics_model) > 0.90