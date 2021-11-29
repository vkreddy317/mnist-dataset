from flask import Flask
from flask import request
from joblib import load
import os
import numpy as np
app = Flask(__name__)

best_model_folder_1 = '/exp/mnist/models/test_0.15_val_0.15_hyperparameter_0.001_i_2/models.joblib'
clf1 = load(best_model_folder_1)

best_model_folder_2 = '/exp/mnist/models/test_0.15_val_0.15_hyperparameter_8_i_2/models.joblib'
clf2 = load(best_model_folder_2)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/decision_tree_predict", methods=['POST'])
def predict_1():
        input_json = request.json
        image = input_json['image']
        print(image)
        image = np.array(image).reshape(1,-1)
        predicted = clf2.predict(image)
        return str(predicted[0])

@app.route("/svm_predict", methods=['POST'])
def predict_2():
        input_json = request.json
        image = input_json['image']
        print(image)
        image = np.array(image).reshape(1,-1)
        predicted = clf1.predict(image)
        return str(predicted[0])

if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8000)
