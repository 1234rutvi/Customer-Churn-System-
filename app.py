from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model safely
model_path = os.path.join("model", "churn_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")   # FIXED

@app.route("/predict", methods=["POST"])
def predict():
    try:
        tenure = float(request.form["tenure"])
        monthly_charges = float(request.form["monthly_charges"])
        
        features = np.array([[tenure, monthly_charges]])
        prediction = model.predict(features)

        result = "Customer will churn" if prediction[0] == 1 else "Customer will stay"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == "__main__":
    app.run()
