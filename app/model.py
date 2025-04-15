import joblib
import numpy as np
from datetime import timedelta
from tensorflow.keras.models import load_model as load_keras_model
from app.preprocess import update_sequence
import keras
import tensorflow as tf
from keras import config

# Autoriser explicitement la désérialisation de lambda
config.enable_unsafe_deserialization()


def softmax_over_time(x):
    return tf.nn.softmax(x, axis=1)

def sum_context_vector(x):
    return tf.reduce_sum(x, axis=1)


def load_model():
    custom_objects = {
        'softmax_over_time': softmax_over_time,
        'sum_context_vector': sum_context_vector
    }
    # Charger le modèle avec les objets personnalisés
    model = load_keras_model(
        "./model/modele_consommation.keras",
        custom_objects=custom_objects,
        compile=False
    )
    scaler = joblib.load("./model/scaler.joblib")
    features = joblib.load("./model/features.joblib")
    target_index = features.index("Consommation brute électricité")
    return model, scaler, features, target_index

def make_prediction(start_date, n_steps, model, scaler, features, target_index):
    # Logique de prédiction
    last_seq = joblib.load("./model/last_sequences.joblib")
    future_preds = []
    
    for i in range(n_steps):
        current_time = start_date + timedelta(minutes=30*i)
        pred = model.predict(np.array([last_seq]), verbose=0)[0][0]
        future_preds.append(pred)
        
        new_row = update_sequence(last_seq, current_time, pred, features, target_index)
        last_seq = np.vstack([last_seq[1:], new_row])
    
    temp = np.zeros((len(future_preds), len(features)))
    temp[:, target_index] = future_preds
    future_denorm = scaler.inverse_transform(temp)[:, target_index]
    
    return future_denorm.tolist()