from flask import Flask, render_template, request
import numpy as np
import joblib
import os

# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None
    input_data = {}

    if request.method == "POST":
        try:
            # Get user input
            input_data = {
                "Pregnancies": float(request.form["Pregnancies"]),
                "Glucose": float(request.form["Glucose"]),
                "BloodPressure": float(request.form["BloodPressure"]),
                "SkinThickness": float(request.form["SkinThickness"]),
                "Insulin": float(request.form["Insulin"]),
                "BMI": float(request.form["BMI"]),
                "DiabetesPedigreeFunction": float(request.form["DiabetesPedigreeFunction"]),
                "Age": float(request.form["Age"]),
            }

            features = list(input_data.values())
            final_input = scaler.transform([features])

            prediction = model.predict(final_input)[0]
            probability = model.predict_proba(final_input)[0][1]  # probability of class 1

            result = "Diabetic" if prediction == 1 else "Not Diabetic"

        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template("index.html", result=result, input_data=input_data, probability=probability)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
