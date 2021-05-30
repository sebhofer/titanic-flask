import os

from flask import (
    Flask, flash, g, redirect, render_template, request, session, url_for
)

app = Flask(__name__, template_folder="templates")

@app.route('/')
def home():
    return render_template("predict.html", prediction_text = "")

@app.route('/predict', methods=('GET', 'POST'))
def predict():
    if request.method == 'POST':
        # do validation here
        return render_template("predict.html", prediction_text = request.form['prediction'])
    else:
        return home()