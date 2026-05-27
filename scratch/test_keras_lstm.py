# %%
import os
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input

print("Keras version:", keras.__version__)
print("Backend:", keras.config.backend())

# Create dummy 3D sequence data
# (samples, lookback, features)
X_dummy = np.random.randn(100, 60, 16)
y_dummy = np.random.randn(100)

# Build stacked LSTM model
model = Sequential([
    Input(shape=(60, 16)),
    LSTM(units=32, return_sequences=True),
    Dropout(0.2),
    LSTM(units=32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Train for 1 epoch
model.fit(X_dummy, y_dummy, epochs=1, batch_size=32, verbose=1)

# Save the model to .keras format
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "test_lstm_model.keras")
model.save(model_path)
print("Model saved to:", model_path)

# Try loading the model back
loaded_model = keras.models.load_model(model_path)
print("Model loaded successfully!")
