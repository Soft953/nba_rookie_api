from flask import Flask
app = Flask(__name__)


# Simple Flask API

@app.route('/')
def hello_world():
    return 'Hello, World!'

