from flask import Flask, render_template, request
import pickle
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Load model
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

# You can manually set accuracy (since dataset not on Render)
accuracy = 85.0   # Change to your real accuracy

@app.route("/")
def home():
    return render_template("index.html", accuracy=accuracy)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        tenure = request.form.get("tenure")
        monthly_charges = request.form.get("monthly_charges")

        # Input validation
        if not tenure or not monthly_charges:
            return render_template("index.html",
                                   error="Please fill all fields.",
                                   accuracy=accuracy)

        tenure = float(tenure)
        monthly_charges = float(monthly_charges)

        if tenure < 0 or monthly_charges < 0:
            return render_template("index.html",
                                   error="Values cannot be negative.",
                                   accuracy=accuracy)

        features = np.array([[tenure, monthly_charges]])

        prediction = model.predict(features)[0]
        probability = round(model.predict_proba(features)[0][1] * 100, 2)

        result = "Customer will churn" if prediction == 1 else "Customer will stay"

        # ---- Decision Boundary Plot ----
        x_min, x_max = 0, 72
        y_min, y_max = 0, 150

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100),
            np.linspace(y_min, y_max, 100)
        )

        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(grid)
        Z = Z.reshape(xx.shape)

        plt.figure()
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(tenure, monthly_charges)
        plt.xlabel("Tenure")
        plt.ylabel("Monthly Charges")

        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return render_template("index.html",
                               prediction_text=result,
                               probability=probability,
                               plot_url=plot_url,
                               accuracy=accuracy)

    except Exception:
        return render_template("index.html",
                               error="Something went wrong.",
                               accuracy=accuracy)

if __name__ == "__main__":
    app.run()
