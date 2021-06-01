# titanic-flask

Simple web interface to estimate survival probabilities for the Titanic desaster.

To run first do (from project root):
```
$ conda env create --file environment.yml
$ python ./src/model.py
$ python ./src/app.py
```
or
```
$ docker pull sebhofer/titanic-flask
$ docker run -d -p 5000:5000 sebhofer/titanic-flask
```
then open `localhost:5000/predict` in your browser.
