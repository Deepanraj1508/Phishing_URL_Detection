from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pickle
from feature import FeatureExtraction

app = Flask(__name__)

# Load model
with open("model/model.pkl", "rb") as f:
    gbc = pickle.load(f)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    url = request.form.get("url")

    obj = FeatureExtraction(url)
    x = np.array(obj.getFeaturesList()).reshape(1, 30)

    prob_safe = gbc.predict_proba(x)[0][1]  # probability safe

    return render_template("result.html", url=url, prob=round(prob_safe, 2))


if __name__ == "__main__":
    app.run(debug=True)
