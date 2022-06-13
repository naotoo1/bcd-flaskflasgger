import pandas as pd
import streamlit as st
from prosemble import Hybrid
import numpy as np
import pickle

# 2. Create the app object
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


def predict_BreastCancer(radius_mean, radius_texture, method):
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
    # prediction using the svc,knn and dtc models
    pred1 = svc.predict([[radius_mean, radius_texture]])
    pred2 = knn.predict([[radius_mean, radius_texture]])
    pred3 = dtc.predict([[radius_mean, radius_texture]])

    # confidence of prediction using the svc,knn and dtc models respectively
    sec1 = get_posterior(x=[[radius_mean, radius_texture]], y_=pred1, z_=svc)
    sec2 = get_posterior(x=[[radius_mean, radius_texture]], y_=pred2, z_=knn)
    sec3 = get_posterior(x=[[radius_mean, radius_texture]], y_=pred3, z_=dtc)
    all_pred = [pred1, pred2, pred3]
    all_sec = [sec1, sec2, sec3]
    # prediction from the ensemble using hard voting
    prediction1 = ensemble.pred_prob([[radius_mean, radius_texture]], all_pred)
    # prediction from the ensemble using soft voting
    prediction2 = ensemble.pred_sprob([[radius_mean, radius_texture]], all_sec)
    # confidence of the prediction using hard voting
    hard_prob = ensemble.prob([[radius_mean, radius_texture]], all_pred)
    # confidence of the prediction using soft voting
    soft_prob = ensemble.sprob([[radius_mean, radius_texture]], all_sec)
    if method == 'soft':
        if prediction2[0] > 0.5:
            return f"Benign with {round(soft_prob[0] * 100, 2)}% confidence"
        else:
            return f"Malignant with {round(soft_prob[0] * 100, 2)}% confidence"

    if method == 'hard':
        if prediction1[0] > 0.5:
            return f"Benign with {round(hard_prob[0] * 100, 2)}% confidence"
        else:
            return f"Malignant with {round(hard_prob[0] * 100, 2)}% confidence"

        
def main():
    st.title(" Breast Cancer Diagnostic Tool")
    html_temp = """
        <div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;">Streamlit Breast Cancer Diagnostic ML App </h2>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)
    Radius_mean = st.text_input("Radius_mean", "Type Here")
    Radius_texture = st.text_input("Radius_texture", "Type Here")
    Method = st.text_input("Method", "Type Here")
    pred = ""
    if st.button("Predict"):
        pred = predict_BreastCancer(radius_mean=Radius_mean, radius_texture=Radius_texture, method=Method)
    st.success('The output is {}'.format(pred))
    if st.button("About"):
        st.text("This Breast Cancer Diagnostic ML App is built with Streamlit")
        st.text("Built with Streamlit")


if __name__ == '__main__':
    main()

# run streamlit app.py
