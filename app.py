from flask import Flask
from flask import request
import boto3


app = Flask(__name__)


@app.route("/")
def index():
    return "Hello World!"


if __name__ == "__main__":
    app.run(debug=True, port=5000)
