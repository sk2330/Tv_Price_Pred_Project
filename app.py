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

# Load the encoder categories from .pkl files
with open('Transformation/onehot_Brand_encoder.pkl', 'rb') as f:
    ohe_Brand.categories_ = pickle.load(f)
    
with open('Transformation/onehot_Screen_encoder.pkl', 'rb') as f:
    ohe_Screen.categories_ = pickle.load(f)
    
with open('Transformation/onehot_Display_encoder.pkl', 'rb') as f:
    ohe_Display.categories_ = pickle.load(f)
    
with open('Transformation/onehot_Platform_encoder.pkl', 'rb') as f:
    ohe_Platform.categories_ = pickle.load(f)

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
        features = np.hstack((
            encoded_Brand,
            encoded_Screen,
            encoded_Display,
            encoded_Platform,
            np.array([[Inches]])
        ))

        # Make the prediction
        prediction = model.predict(features)
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text="The estimated TV price is Rs. {}".format(output))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
