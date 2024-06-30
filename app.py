from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


app = Flask(__name__)
model = pickle.load(open("Tv_price_xgb.pkl", "rb"))

# Initialize the OneHotEncoders
ohe_Brand = OneHotEncoder(handle_unknown='ignore')
ohe_Screen = OneHotEncoder(handle_unknown='ignore')
ohe_Display = OneHotEncoder(handle_unknown='ignore')
ohe_Platform = OneHotEncoder(handle_unknown='ignore')

ohe_Brand.categories_ = np.load('Transformation\onehot_Brand_encoder.npy', allow_pickle=True)
ohe_Screen.categories_ = np.load('Transformation\onehot_Screen_encoder.npy', allow_pickle=True)
ohe_Display.categories_ = np.load('Transformation\onehot_Display_encoder.npy', allow_pickle=True)
ohe_Platform.categories_ = np.load('Transformation\onehot_Platform_encoder.npy', allow_pickle=True)

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
        encoded_Brand = ohe_Brand.transform([[Brand]]).toarray()
        encoded_Screen = ohe_Screen.transform([[Screen]]).toarray()
        encoded_Display = ohe_Display.transform([[Display]]).toarray()
        encoded_Platform = ohe_Platform.transform([[Platform]]).toarray()

        # Combine the encoded features with the numerical features
        # features = np.hstack((
        #     encoded_Brand,
        #     encoded_Screen,
        #     encoded_Display,
        #     encoded_Platform,
        #     np.array([[Inches]])
        # ))

        # Make the prediction
        prediction = model.predict([[
            encoded_Brand[0][0], encoded_Brand[0][1], encoded_Brand[0][2], encoded_Brand[0][3], encoded_Brand[0][4],
            encoded_Screen[0][0], encoded_Screen[0][1], encoded_Screen[0][2], encoded_Screen[0][3], encoded_Screen[0][4],
            encoded_Display[0][0], encoded_Display[0][1], encoded_Display[0][2], encoded_Display[0][3], encoded_Display[0][4],
            encoded_Platform[0][0], encoded_Platform[0][1], encoded_Platform[0][2], encoded_Platform[0][3], encoded_Platform[0][4],
            Inches
        ]])
        
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text="The estimated TV price is Rs. {}".format(output))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
