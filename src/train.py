import pandas as pd
import tensorflow as tf
from src.window_generator import WindowGenerator
from src.model import build_lstm_model

# 1. Load Data (Using a standard climatic dataset for realism)
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)

df = pd.read_csv(csv_path)
# Simplified: Use just Temperature (T (degC)) and Pressure (p (mbar)) to predict Temperature
df = df[5::6] # Subsample from 10 min to 1 hour
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
lstm_model = build_lstm_model(input_shape=(24, 2)) # 24 steps, 2 features
history = lstm_model.fit(window.train, epochs=10, validation_data=window.val)

# 6. Save Model
lstm_model.save('forecasting_engine.h5')
print("✅ Model trained and saved as forecasting_engine.h5")