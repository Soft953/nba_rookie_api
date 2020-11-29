from flask import Flask

from model import Model

app = Flask(__name__)


# Simple Flask API

@app.route('/')
def hello_world():
    try:
      clf = Model('models/clf_logistic.joblib')
      return str(clf)
    except ValueError as e:
      return 'Error ' + str(e)
    return 'Hello, World! - try to fix  !     No default language could be detected for this app.'

