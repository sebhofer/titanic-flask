import os

from flask import Flask, flash, g, redirect, render_template, request, session, url_for
from create_model import predict_proba, load_model

app = Flask(__name__, template_folder="templates")


@app.route("/")
def home():
    return render_template("predict.html", prediction_text="")


@app.route("/predict", methods=("GET", "POST"))
def predict():
    if request.method == "POST":
        # throws if inputs can't be converted
        age_float = float(request.form["age"])
        pclass_int = int(request.form["pclass"])
        sex = request.form["sex"]
        model, transformer = load_model()
        prediction = predict_proba(sex, age_float, pclass_int, transformer, model)
        return render_template("predict.html", prediction_text=prediction)
    else:
        return home()
