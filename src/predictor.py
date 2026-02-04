import numpy as np
import tensorflow as tf
import joblib
import os

BASE = os.path.dirname(__file__)
MODEL = os.path.join(BASE, "../models/kone_production_lstm.keras")
CONF  = os.path.join(BASE, "../models/production_config.joblib")

def download_if_missing():
    os.makedirs(os.path.join(BASE, "../models"), exist_ok=True)

    if not os.path.exists(MODEL):
        gdown.download("https://drive.google.com/file/d/1JeoYHNNBMHwQcZgSxZQ_uQjBV0jV-Ogj/view?usp=drive_link", MODEL, quiet=False)

    if not os.path.exists(CONF):
        gdown.download("https://drive.google.com/file/d/1u7JWHykAk5dSKZK1B2DhBifj6SpIuvlS/view?usp=drive_link", CONF, quiet=False)

download_if_missing()

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        pt = tf.exp(-bce)
        return alpha * (1 - pt) ** gamma * bce
    return focal_loss_fixed

model = tf.keras.models.load_model(
    MODEL,
    custom_objects={'focal_loss_fixed': focal_loss()}
)

config = joblib.load(CONF)
THRESH = config["threshold"]

history = []

def predict_live(row):

    global history

    vec = np.array([
        row["vibration"],
        row["usage_rate"],
        row["door_cycles"],
        row["speed"],
        row["temperature"],
        row["vibration"],
        row["door_cycles"],
        row["temperature"]
    ])

    history.append(vec)
    if len(history) > 10:
        history.pop(0)

    while len(history) < 10:
        history.insert(0, vec)

    seq = np.array(history).reshape(1,10,8)

    proba = model.predict(seq, verbose=0)[0][0]

    return proba, proba > THRESH
