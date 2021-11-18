from flask import Flask
from flask import request
from flask_restx import Resource, Api
from werkzeug.utils import cached_property
import numpy as np
import joblib


best_model = '../models/tt_0.2_val_0.1_round_0_svm_hyper_0.001/model.joblib'

app = Flask(__name__)
api = Api(app)

#@app.route("/")
#def hello_world():
#    return "<p> Hello World!</p>"
@api.route('/hello')
class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


@app.route("/predict",methods=['POST'])
def predict():
    model = joblib.load(best_model)
    input_json = request.json
    image = input_json['image']
    image = np.array(image).reshape(1,-1)
    predict = model.predict(image)
    print("Iamge Sent :- ",image)
    print(str(predict[0]))
    return str(predict[0])

if __name__=='__main__':
    #app.run(debug=True)
    app.run()