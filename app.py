from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model/churn_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("templates/index.html")

@app.route("/predict", methods=["POST"])
def predict():
    tenure = float(request.form["tenure"])
    monthly_charges = float(request.form["monthly_charges"])
    
    features = np.array([[tenure, monthly_charges]])
    prediction = model.predict(features)

    result = "Customer will churn" if prediction[0] == 1 else "Customer will stay"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
