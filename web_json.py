from flask import Flask, request, jsonify

import sys

application = Flask(__name__)


@application.route("/hello")
def hello():
    data = request.args.to_dict()
    return jsonify(data)



if __name__ == "__main__":
    application.run(host='localhost', port=8080)