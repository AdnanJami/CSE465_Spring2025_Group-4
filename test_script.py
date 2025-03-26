from joblib import load
import tensorflow as tf
import numpy as np
import pandas as pd

df = pd.read_csv("./Augmented_data/augmented_testset.csv")[:100]

# Define features and target
X = df.iloc[:, 1:].values
Y = df.iloc[:, 0].values

# Load artifacts
model = tf.keras.models.load_model('./Model_file/best_model.h5')
scaler = load('./Model_file/scaler.joblib')
label_encoder = load('./Model_file/label_encoder.joblib')
count = 0
for i,x in enumerate(X):
    print(x)
    # Prepare new data (example)
    new_data = x  # Your input features
    scaled_data = scaler.transform([x])
    
    # Make prediction
    prediction = model.predict(scaled_data)
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
    count += 1 if predicted_class[0] == Y[i] else 0
    print(f"Predicted class: {predicted_class[0]}")
print("count: ",count)
print("total: ",len(df))
print("Accuracy: ",(count/len(df))*100)
