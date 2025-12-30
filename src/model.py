import tensorflow as tf
from tensorflow.keras import layers, models

def build_lstm_model(input_shape):
    model = models.Sequential([
        # LSTM Layer 1: Returns sequences to feed into the next LSTM layer
        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2), # Prevent overfitting
        
        # LSTM Layer 2: Compresses info into a single vector
        layers.LSTM(32, return_sequences=False),
        
        # Dense Output Layer: Predicts the value
        layers.Dense(1)
    ])
    
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    return model