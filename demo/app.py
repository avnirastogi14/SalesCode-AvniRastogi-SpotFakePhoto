# from flask import Flask, render_template, request, jsonify
# import cv2
# import numpy as np
# import tempfile
# import os

# import sys
# from pathlib import Path

# ROOT = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(ROOT))

# from predict import predict

# app = Flask(__name__)


# @app.route("/")
# def index():
#     return render_template("index.html")


# @app.route("/predict", methods=["POST"])
# def runPrediction():

#     file = request.files["image"]

#     temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")

#     file.save(temp.name)

#     score = predict(temp.name)

#     os.remove(temp.name)

#     return jsonify({
#         "score": float(score)
#     })

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import tempfile
import os

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from predict import predict

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def runPrediction():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp.close()

    try:
        file.save(temp.name)
        score = predict(temp.name)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp.name):
            os.remove(temp.name)

    return jsonify({
        "score": float(score)
    })

if __name__ == "__main__":
    app.run(debug=True)