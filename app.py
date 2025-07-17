from flask import Flask, render_template, request
import numpy as np
import joblib
import os

# Load the model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler-for-render.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            # Collect input features from form
            features = [
                float(request.form["Pregnancies"]),
                float(request.form["Glucose"]),
                float(request.form["BloodPressure"]),
                float(request.form["SkinThickness"]),
                float(request.form["Insulin"]),
                float(request.form["BMI"]),
                float(request.form["DiabetesPedigreeFunction"]),
                float(request.form["Age"])
            ]

            # Preprocess and predict
            input_data = scaler.transform([features])
            prediction = model.predict(input_data)[0]
            result = "Diabetic" if prediction == 1 else "Not Diabetic"
        except:
            result = "Invalid input. Please enter valid numeric values."

    return render_template("index.html", result=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
