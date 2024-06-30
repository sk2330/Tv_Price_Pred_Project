from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)
model = pickle.load(open("Tv_price_xgb.pkl", "rb"))

# Load the OneHotEncoders
ohe_Brand = OneHotEncoder(handle_unknown='ignore')
ohe_Screen = OneHotEncoder(handle_unknown='ignore')
ohe_Display = OneHotEncoder(handle_unknown='ignore')
ohe_Platform = OneHotEncoder(handle_unknown='ignore')

ohe_Brand.categories_ = np.load('Transformation/onehot_Brand_encoder.npy', allow_pickle=True)
ohe_Screen.categories_ = np.load('Transformation/onehot_Screen_encoder.npy', allow_pickle=True)
ohe_Display.categories_ = np.load('Transformation/ohe_Display_encoder.npy', allow_pickle=True)
ohe_Platform.categories_ = np.load('Transformation/ohe_Platform_encoder.npy', allow_pickle=True)

@app.route("/")
@cross_origin()
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        # Extracting features from the form
        Brand = request.form["Brand"]
        Inches = int(request.form["Inches"])
        Screen = request.form["Screen"]
        Display = request.form["Display"]
        Platform = request.form["Platform"]

        # OneHot encode the categorical variables
        encoded_Brand = ohe_Brand.transform([[Brand]])
        encoded_Screen = ohe_Screen.transform([[Screen]])
        encoded_Display = ohe_Display.transform([[Display]])
        encoded_Platform = ohe_Platform.transform([[Platform]])

        # Combine the encoded features with the numerical features
        features = np.hstack([
            encoded_Brand,
            encoded_Screen,
            encoded_Display,
            encoded_Platform,
            np.array([[Inches]])
        ]).toarray()

        # Make the prediction
        prediction = model.predict(features)
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text="The estimated TV price is Rs. {}".format(output))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
