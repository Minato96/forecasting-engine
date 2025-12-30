import pandas as pd
import tensorflow as tf
import os
from src.window_generator import WindowGenerator
from src.model import build_lstm_model

# 1. Load Data
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)

# --- THE FIX IS HERE ---
# Instead of guessing, we build the path explicitly.
# The file inside the zip is always named 'jena_climate_2009_2016.csv'
csv_path = os.path.join(os.path.dirname(zip_path), 'jena_climate_2009_2016.csv')

print(f"📂 Loading data from: {csv_path}")
df = pd.read_csv(csv_path)

# Subsample from 10 min to 1 hour (take every 6th row)
df = df[5::6] 

# Use just Temperature (T (degC)) and Pressure (p (mbar))
df = df[['p (mbar)', 'T (degC)']] 

# 2. Split Data
n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

# 3. Normalize (CRITICAL for Neural Networks)
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# 4. Create Window Generator
# Input: Past 24 hours. Output: Next 1 hour.
window = WindowGenerator(input_width=24, label_width=1, shift=1,
                         train_df=train_df, val_df=val_df, test_df=test_df,
                         label_columns=['T (degC)'])

# 5. Build & Train
# 24 steps, 2 features (Temp + Pressure)
lstm_model = build_lstm_model(input_shape=(24, 2)) 

print("🚀 Starting Training...")
history = lstm_model.fit(window.train, epochs=10, validation_data=window.val)

# 6. Save Model
lstm_model.save('forecasting_engine.h5')
print("✅ Model trained and saved as forecasting_engine.h5")