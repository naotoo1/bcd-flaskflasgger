import pandas as pd
from flask import Flask, request
from flasgger import Swagger
from prosemble import Hybrid
import numpy as np
import pickle

# 2. Create the app object
app = Flask(__name__)
Swagger(app)
pickle_in1 = open("svc.pkl", "rb")
pickle_in2 = open("knn.pkl", "rb")
pickle_in3 = open("dtc.pkl", "rb")

svc = pickle.load(pickle_in1)
knn = pickle.load(pickle_in2)
dtc = pickle.load(pickle_in3)


def get_posterior(x, y_, z_):
    """

    :param x: Input data
    :param y_: prediction
    :param z_: model
    :return: prediction probabilities
    """
    z1 = z_.predict_proba(x)
    certainties = [np.max(i) for i in z1]
    cert = np.array(certainties).flatten()
    cert = cert.reshape(len(cert), 1)
    y_ = y_.reshape(len(y_), 1)
    labels_with_certainty = np.concatenate((y_, cert), axis=1)
    return np.round(labels_with_certainty, 4)


def result(x, y, z):
    """

    :param x: predicted labels from the ensemble
    :param y: class label of Malignant
    :param z: confidence of the predicted labels
    :return: predicted labels with corresponding confidence
    """
    result1 = []
    c = -1
    for i in x:
        c += 1
        if i == y:
            result1.append(f"Malignant with {round(z[c] * 100, 2)}% confidence")
        else:
            result1.append(f"Benign with {round(z[c] * 100, 2)}% confidence")
    return str(list(result1))


# classes labels
proto_classes = np.array([0, 1])

# object of Hybrid class from prosemble
ensemble = Hybrid(model_prototypes=None, proto_classes=proto_classes, mm=2, omega_matrix=None, matrix='n')


@app.route('/')
def welcome():
    return " welcome to bcd-Flask"


@app.route('/predict', methods=["Get"])
def predict_BreastCancer():
    """Diagnose the breast cancer
    ---
    parameters:
      - name: Radius_mean
        in: query
        type: number
        required: true
      - name: Radius_texture
        in: query
        type: number
        required: true
      - name: Method
        in: query
        type: string
        required: true
    responses:
        200:
            description: The output values

    """
    Radius_mean = request.args.get('Radius_mean')
    Radius_texture = request.args.get('Radius_texture')
    Method = request.args.get('Method')

    # prediction using the svc,knn and dtc models
    pred1 = svc.predict([[Radius_mean, Radius_texture]])
    pred2 = knn.predict([[Radius_mean, Radius_texture]])
    pred3 = dtc.predict([[Radius_mean, Radius_texture]])

    # confidence of prediction using the svc,knn and dtc models respectively
    sec1 = get_posterior(x=[[Radius_mean, Radius_texture]], y_=pred1, z_=svc)
    sec2 = get_posterior(x=[[Radius_mean, Radius_texture]], y_=pred2, z_=knn)
    sec3 = get_posterior(x=[[Radius_mean, Radius_texture]], y_=pred3, z_=dtc)
    all_pred = [pred1, pred2, pred3]
    all_sec = [sec1, sec2, sec3]
    # prediction from the ensemble using hard voting
    prediction1 = ensemble.pred_prob([[Radius_mean, Radius_texture]], all_pred)
    # prediction from the ensemble using soft voting
    prediction2 = ensemble.pred_sprob([[Radius_mean, Radius_texture]], all_sec)
    # confidence of the prediction using hard voting
    hard_prob = ensemble.prob([[Radius_mean, Radius_texture]], all_pred)
    # confidence of the prediction using soft voting
    soft_prob = ensemble.sprob([[Radius_mean, Radius_texture]], all_sec)
    if Method == 'soft':
        if prediction2[0] > 0.5:
            return f"Benign with {round(soft_prob[0] * 100, 2)}% confidence"
        else:
            return f"Malignant with {round(soft_prob[0] * 100, 2)}% confidence"

    if Method == 'hard':
        if prediction1[0] > 0.5:
            return f"Benign with {round(hard_prob[0] * 100, 2)}% confidence"
        else:
            return f"Malignant with {round(hard_prob[0] * 100, 2)}% confidence"


@app.route('/predict_file', methods=['POST'])
def predict_file():
    """
    Diagnose breast cancer using data_in_file.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      - name: Method
        in: query
        type: string
        required: true

    responses:
        200:
            description: The output values
    """
    df = pd.read_csv(request.files.get('file'))
    Method = request.args.get('Method')
    print((df.head()))

    # prediction using the svc,knn and dtc models
    pred1 = svc.predict(df)
    pred2 = knn.predict(df)
    pred3 = dtc.predict(df)

    # confidence of prediction using the svc,knn and dtc models respectively
    sec1 = get_posterior(x=df, y_=pred1, z_=svc)
    sec2 = get_posterior(x=df, y_=pred2, z_=knn)
    sec3 = get_posterior(x=df, y_=pred3, z_=dtc)
    all_pred = [pred1, pred2, pred3]
    all_sec = [sec1, sec2, sec3]

    # prediction from the ensemble using hard voting
    prediction1 = ensemble.pred_prob(df, all_pred)
    # prediction from the ensemble using soft voting
    prediction2 = ensemble.pred_sprob(df, all_sec)
    # confidence of the prediction using hard voting
    hard_prob = ensemble.prob(df, all_pred)
    # confidence of the prediction using soft voting
    soft_prob = ensemble.sprob(df, all_sec)
    if Method == 'soft':
        return result(x=prediction2, y=0, z=soft_prob)
    if Method == 'hard':
        return result(x=prediction1, y=0, z=hard_prob)


if __name__ == '__main__':
    app.run()

# visit http://localhost/tool to open the PyWebIO application.
