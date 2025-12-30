from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import uvicorn

app = FastAPI(title="Neural Forecasting Engine")

# Load model once at startup
model = tf.keras.models.load_model('forecasting_engine.h5')

class InputData(BaseModel):
    # Expects a list of 24 time steps, each with 2 features [Pressure, Temp]
    history: list[list[float]] 

@app.post("/predict")
def predict_next_hour(data: InputData):
    # Convert input list to numpy array: shape (1, 24, 2)
    input_array = np.array(data.history).reshape(1, 24, 2)
    
    # Inference
    prediction = model.predict(input_array)
    
    # Return result (un-normalize if this was real prod code)
    return {"predicted_temperature_normalized": float(prediction[0][0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)