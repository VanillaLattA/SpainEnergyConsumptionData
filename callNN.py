import os
from flask import Flask, jsonify, request, Response, send_file
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import LeakyReLU
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import shap
import pandas as pd

model = keras.models.load_model('LoadNN.ann')

nndata = pd.read_csv("NNData.csv")
nndata.drop(["load"], inplace=True, axis=1)

scales = np.asarray([23.0, 40.115214679859605, 79.76803501358285, 96.34228795653486, 20.70917597343797, 22974.0], np.float64)
mins = np.asarray([0.0, 269.77175332496984, 19.867491699366134, 0.0, 0.0, 18041.0], np.float64)

row = nndata.iloc[0]
row = pd.DataFrame({
    "Time": [row["time"]],
    "Temperature": [row["tAve"]],
    "Humidity": [row["hAve"]],
    "Cloud Cover": [row["cAve"]],
    "Wind Speed": [row["wAve"]]
})

y = model.predict(row.to_numpy(), verbose=0)[0][0]

explainer = shap.DeepExplainer(model, data=nndata.to_numpy())

shap_values = explainer.shap_values(
    row.to_numpy(), check_additivity=True
)

for i in range(5):
    shap_values[0][0][i] *= scales[5]
    row.iloc[0, i] *= scales[i]
    row.iloc[0, i] += mins[i]

ev = explainer.expected_value[0].numpy()
ev *= scales[5]
ev += mins[5]
row = row.round(decimals = 2)

fig = plt.gcf()
fig = shap.force_plot(
    ev,
    shap_values[0][0],
    row,
    text_rotation=70,
    show=False,
    matplotlib=True,
    contribution_threshold=0
)
fig.set_size_inches(20, 5)
fig.set_facecolor('#FCF9D9')
fig.axes[0].set_facecolor('#FCF9D9')
plt.savefig("public/shap_nn.png")

