import sqlite3 as sql
import numpy as np
import pandas as pd
import pickle
import os
import joblib
import onnx
import onnxruntime as rt
import torch

filename = "./svm_iris.onnx"


PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
rf_model_loaded = onnx.load(os.path.join(PROJECT_ROOT, "static/rf_model_init.onnx"))
sess = rt.InferenceSession(os.path.join(PROJECT_ROOT, "static/rf_model_init.onnx"))
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

# scaler_loaded = onnx.load(os.path.join(PROJECT_ROOT, "static/scaler_init.onnx"))
# onnx.checker.check_model(scaler_loaded)

fixed_acidity = 7.5
volatile_acidity = 0.75
citric_acid = 3.00
residual_sugar = 3.9
chlorides = 0.176
free_sulfur_dioxide = 12.0
total_sulfur_dioxide = 35.0
density = 0.91
ph = 3.9
sulphates = 0.56
alcohol = 9.4

X = np.array(
    [
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        ph,
        sulphates,
        alcohol,
    ]
)
pred_onx = sess.run(None, {input_name: X.astype(np.float32)})[0]
print(pred_onx)

X_scaled = scaler_loaded.transform(X)
quality = rf_model_loaded.predict(X_scaled)[0]

res = "хорошее" if quality == 1 else "плохое"
msg = f"Вино {res}"
