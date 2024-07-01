from flask import Flask, request, render_template
from flask_cors import cross_origin
import joblib
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("Tv_price_xgb.pkl", "rb"))

# Load the OneHotEncoders
ohe_Brand = joblib.load('Pickle_files/onehot_Brand_encoder.pkl')
ohe_Screen = joblib.load('Pickle_files/onehot_Screen_encoder.pkl')
ohe_Display = joblib.load('Pickle_files/onehot_Display_encoder.pkl')
ohe_Platform = joblib.load('Pickle_files/onehot_Platform_encoder.pkl')
ohe_Model_number= joblib.load('Pickle_files/onehot_Model_Number_encoder.pkl')

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
        Model_Number = request.form["Model_Number"]

        # OneHot encode the categorical variables
        encoded_Brand = ohe_Brand.transform([[Brand]])
        encoded_Screen = ohe_Screen.transform([[Screen]])
        encoded_Display = ohe_Display.transform([[Display]])
        encoded_Platform = ohe_Platform.transform([[Platform]])
        encoded_Model_number = ohe_Model_number.transform([[Model_Number]])

        # Combine the encoded features with the numerical features
        features = np.hstack((
            encoded_Brand,
            encoded_Screen,
            encoded_Display,
            encoded_Platform,
            encoded_Model_number,
            np.array([[Inches]])
        ))

        # Make the prediction
        prediction = model.predict(features)
        
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text="The estimated TV price is Rs. {}".format(output))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
