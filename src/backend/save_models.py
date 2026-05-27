"""
Model Saving and Loading Utilities.

Provides functions to save and load ML models using pickle, joblib,
and Keras formats. TensorFlow/Keras is a required dependency.

Author: Stock Market AI Advisor Team
"""

import os
import sys
import pickle
import logging

import joblib

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

logger = logging.getLogger(__name__)


def save_model_pickle(model, filepath):
    """Save a model using pickle serialization."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved (pickle): {filepath}")


def load_model_pickle(filepath):
    """Load a model from a pickle file."""
    if not os.path.exists(filepath):
        logger.warning(f"Model file not found: {filepath}")
        return None
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Model loaded (pickle): {filepath}")
    return model


def save_model_joblib(model, filepath):
    """Save a model using joblib serialization."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    logger.info(f"Model saved (joblib): {filepath}")


def load_model_joblib(filepath):
    """Load a model from a joblib file."""
    if not os.path.exists(filepath):
        logger.warning(f"Model file not found: {filepath}")
        return None
    model = joblib.load(filepath)
    logger.info(f"Model loaded (joblib): {filepath}")
    return model


def save_keras_model(model, filepath):
    """Save a Keras model in .keras format."""
    if not HAS_TENSORFLOW:
        logger.warning("TensorFlow not installed. Cannot save Keras model.")
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save(filepath)
    logger.info(f"Keras model saved: {filepath}")


def load_keras_model(filepath):
    """Load a Keras model from .keras or .h5 file."""
    if not HAS_TENSORFLOW:
        logger.warning("TensorFlow not installed. Cannot load Keras model.")
        return None
    if not os.path.exists(filepath):
        logger.warning(f"Keras model file not found: {filepath}")
        return None
    model = keras.models.load_model(filepath)
    logger.info(f"Keras model loaded: {filepath}")
    return model


def save_scaler(scaler, filepath):
    """Save a scaler object using pickle."""
    save_model_pickle(scaler, filepath)


def load_scaler(filepath):
    """Load a scaler object from pickle file."""
    return load_model_pickle(filepath)


def get_model_path(model_name):
    """
    Get the full file path for a model given its short name.

    Args:
        model_name: one of 'random_forest_regressor', 'random_forest_classifier',
                    'xgboost_regressor', 'xgboost_classifier', 'lstm',
                    'user_segment', 'scaler', 'lstm_scaler'

    Returns:
        str: full path to the model file
    """
    model_files = {
        'random_forest_regressor': 'random_forest_regressor.pkl',
        'random_forest_classifier': 'random_forest_classifier.pkl',
        'xgboost_regressor': 'xgboost_regressor.pkl',
        'xgboost_classifier': 'xgboost_classifier.pkl',
        'lstm': 'lstm_model.keras',
        'user_segment': 'user_segment_model.pkl',
        'scaler': 'scaler.pkl',
        'lstm_scaler': 'lstm_scaler.pkl',
        'lstm_feature_scaler': 'lstm_feature_scaler.pkl',
        'lstm_target_scaler': 'lstm_target_scaler.pkl',
    }

    filename = model_files.get(model_name)
    if filename is None:
        logger.warning(f"Unknown model name: {model_name}")
        return None

    return os.path.join(MODEL_DIR, filename)


def list_saved_models():
    """List all saved model files in the models directory."""
    if not os.path.exists(MODEL_DIR):
        return []

    model_files = []
    for f in os.listdir(MODEL_DIR):
        if f.endswith(('.pkl', '.joblib', '.keras', '.h5')):
            full_path = os.path.join(MODEL_DIR, f)
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            model_files.append({
                'name': f,
                'path': full_path,
                'size_mb': round(size_mb, 2),
                'format': os.path.splitext(f)[1]
            })

    return model_files


if __name__ == '__main__':
    print("Model Save/Load Utilities")
    print(f"Model directory: {MODEL_DIR}")
    print(f"TensorFlow available: {HAS_TENSORFLOW}")
    if HAS_TENSORFLOW:
        print(f"TensorFlow version: {tf.__version__}")
    print(f"\nSaved models: {list_saved_models()}")
