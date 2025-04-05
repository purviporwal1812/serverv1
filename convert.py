import tensorflow as tf

# Load your trained Keras model
model = tf.keras.models.load_model('models/final_model.keras')  # or .keras

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model to disk
with open('models/final_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Conversion to TFLite complete!")
