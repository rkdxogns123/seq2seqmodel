from flask import Flask
import sys

application = Flask(__name__)


@application.route("/hello")
def hello():
    return "Hello, World"


if __name__ == "__main__":
    application.run(host='localhost', port=8080)