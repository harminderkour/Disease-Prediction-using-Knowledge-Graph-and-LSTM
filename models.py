# models.py: Load models, scalers, and prediction functions

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
from config import DIABETES_FEATURES, HEART_FEATURES

# Global variables for models and scalers
model_diabetes = None
model_heart = None
scaler_d = StandardScaler()
scaler_h = StandardScaler()

def load_models():
    """Load LSTM models and scalers"""
    global model_diabetes, model_heart, scaler_d, scaler_h
    
    try:
        model_diabetes = load_model("ML/lstm_diabetes_model.h5")
    except Exception:
        model_diabetes = None

    try:
        model_heart = load_model("ML/lstm_heart_model.h5")
    except Exception:
        model_heart = None

    try:
        scaler_d = joblib.load("ML/scaler_diabetes.pkl")
        scaler_h = joblib.load("ML/scaler_heart.pkl")
    except Exception:
        scaler_d = StandardScaler()
        scaler_h = StandardScaler()

def predict_diabetes(user_input_dict):
    """Predict diabetes using LSTM model"""
    if model_diabetes is None:
        return "Model unavailable", 0.0
    
    complete_input = {feature: user_input_dict.get(feature, 0.0) for feature in DIABETES_FEATURES}
    
    X = np.array([[complete_input[f] for f in DIABETES_FEATURES]])
    X_scaled = scaler_d.transform(X)
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    
    pred_probs = model_diabetes.predict(X_lstm, verbose=0)
    pred_class = np.argmax(pred_probs, axis=1)[0]
    confidence = float(pred_probs[0][pred_class])
    
    return "Positive" if pred_class == 1 else "Negative", confidence

def predict_heart(user_input_dict):
    """Predict heart disease using LSTM model"""
    if model_heart is None:
        return "Model unavailable", 0.0
    
    complete_input = {feature: user_input_dict.get(feature, 0.0) for feature in HEART_FEATURES}
    
    X = np.array([[complete_input[f] for f in HEART_FEATURES]])
    X_scaled = scaler_h.transform(X)
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    
    pred_probs = model_heart.predict(X_lstm, verbose=0)
    pred_class = np.argmax(pred_probs, axis=1)[0]
    confidence = float(pred_probs[0][pred_class])
    
    return "High Risk" if pred_class == 1 else "Low Risk", confidence