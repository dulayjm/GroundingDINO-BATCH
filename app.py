import os

from flask import Flask, request
from werkzeug.middleware.proxy_fix import ProxyFix

#from src.inference import load_models, predict_models
import subprocess, os, sys

# special install of the grounding dino work ... 


# result = subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
# print(f"pip install requirements = {result}")

# import subprocess
print('os.getcwd()', os.getcwd())
# 1/0
    # importing the "tarfile" module
#import tarfile

# open file
#file = tarfile.open('model.tar.gz')

# extracting file
#file.extractall(os.getcwd())

#file.close()


# result = subprocess.ru"pip", "install", "gradio==3.27.0"], check=True)
# print(f"pip install gradio==3.27.0 = {result}")

# sys.path.insert(0, "./GroundingDINO")

# result = subprocess.run(["pip", "install", "-e", "./GroundingDINO"], check=True)
# print(f"pip install GroundingDINO = {result}")

from inference import load_model, predict
print('before loading models')
# Load the model by reading the `SM_MODEL_DIR` environment variable
# which is passed to the container by SageMaker (usually /opt/ml/model).
dino_model = load_model()

app = Flask(__name__)
print('after loading models')

# Since the web application runs behind a proxy (nginx), we need to
# add this setting to our app.
app.wsgi_app = ProxyFix(
    app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
)


@app.route("/ping", methods=["GET"])
def ping():
    """
    Healthcheck function.
    """
    return "pong"


@app.route("/invocations", methods=["POST"])
def invocations():
    """
    Function which responds to the invocations requests.
    """
    body = request.json
    return predict(body, dino_model)

if __name__=="__main__":
    app.run(host="hostname", port="80")
