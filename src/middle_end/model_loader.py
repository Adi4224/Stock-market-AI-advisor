"""
Centralized Model Loading with Caching.

Provides a single interface to load all ML models (Random Forest, XGBoost,
LSTM, scalers, user segmentation) with an in-memory cache.

Author: Stock Market AI Advisor Team
"""

import os
import sys
import logging
import pickle

import joblib

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

logger = logging.getLogger(__name__)

# In-memory model cache
_model_cache = {}

# Model file mapping
MODEL_FILES = {
    'random_forest_regressor': ('random_forest_regressor.pkl', 'pkl'),
    'random_forest_classifier': ('random_forest_classifier.pkl', 'pkl'),
    'xgboost_regressor': ('xgboost_regressor.pkl', 'pkl'),
    'xgboost_classifier': ('xgboost_classifier.pkl', 'pkl'),
    'svm_regressor': ('svm_regressor.pkl', 'pkl'),
    'svm_classifier': ('svm_classifier.pkl', 'pkl'),
    'svm_scaler': ('svm_scaler.pkl', 'pkl'),
    'svm_target_scaler': ('svm_target_scaler.pkl', 'pkl'),
    'lstm': ('lstm_model.keras', 'keras'),
    'user_segment': ('user_segment_model.pkl', 'pkl'),
    'scaler': ('scaler.pkl', 'pkl'),
    'lstm_scaler': ('lstm_scaler.pkl', 'pkl'),
    'lstm_feature_scaler': ('lstm_feature_scaler.pkl', 'pkl'),
    'lstm_target_scaler': ('lstm_target_scaler.pkl', 'pkl'),
}


def load_model(model_name):
    """
    Load a model by name with caching.

    Args:
        model_name: one of 'random_forest_regressor', 'random_forest_classifier',
                    'xgboost_regressor', 'xgboost_classifier', 'lstm',
                    'user_segment', 'scaler', 'lstm_scaler'

    Returns:
        Loaded model object, or None if not found
    """
    # Check cache first
    if model_name in _model_cache:
        return _model_cache[model_name]

    if model_name not in MODEL_FILES:
        logger.warning(f"Unknown model name: '{model_name}'. "
                       f"Available: {list(MODEL_FILES.keys())}")
        return None

    filename, fmt = MODEL_FILES[model_name]
    filepath = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(filepath):
        logger.warning(f"Model file not found: {filepath}")
        return None

    try:
        if fmt == 'keras':
            if not HAS_TENSORFLOW:
                logger.warning("TensorFlow not installed. Cannot load LSTM model.")
                return None
            model = keras.models.load_model(filepath)
        elif fmt == 'joblib':
            model = joblib.load(filepath)
        else:  # pkl
            try:
                model = joblib.load(filepath)
            except Exception:
                with open(filepath, 'rb') as f:
                    model = pickle.load(f)

        _model_cache[model_name] = model
        logger.info(f"Model '{model_name}' loaded from {filepath}")
        return model

    except Exception as e:
        logger.error(f"Error loading model '{model_name}': {e}")
        return None


def load_scaler(scaler_name='scaler'):
    """
    Load a scaler by name.

    Args:
        scaler_name: 'scaler', 'lstm_scaler', 'lstm_feature_scaler', or 'lstm_target_scaler'

    Returns:
        Loaded scaler object, or None if not found
    """
    return load_model(scaler_name)


def clear_cache():
    """Clear all cached models from memory."""
    global _model_cache
    _model_cache = {}
    logger.info("Model cache cleared")


def list_available_models():
    """
    List all model files that exist in the models directory.

    Returns:
        list of model names that are available to load
    """
    available = []
    for name, (filename, _) in MODEL_FILES.items():
        filepath = os.path.join(MODEL_DIR, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            available.append({
                'name': name,
                'file': filename,
                'size_mb': round(size_mb, 2)
            })
    return available


def is_model_available(model_name):
    """
    Check if a model file exists.

    Args:
        model_name: name of the model to check

    Returns:
        bool: True if the model file exists
    """
    if model_name not in MODEL_FILES:
        return False
    filename, _ = MODEL_FILES[model_name]
    return os.path.exists(os.path.join(MODEL_DIR, filename))


if __name__ == '__main__':
    print("Model Loader")
    print(f"Model directory: {MODEL_DIR}")
    print(f"\nAvailable models: {list_available_models()}")
