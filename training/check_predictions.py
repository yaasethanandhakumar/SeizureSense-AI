import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

print("Loading test dataset...")

X_test = np.load(
    r"C:\Projects\Epilepsy Seizure Prediction\dataset\X_test.npy"
)

y_test = np.load(
    r"C:\Projects\Epilepsy Seizure Prediction\dataset\y_test.npy"
)

print("Original Shape:", X_test.shape)

# Convert:
# (samples, channels, time)
# to:
# (samples, time, channels)

X_test = np.transpose(X_test, (0, 2, 1))

print("TensorFlow Shape:", X_test.shape)
print("Labels Shape:", y_test.shape)

print("\nLoading trained model...")

model = tf.keras.models.load_model(
    r"C:\Projects\Epilepsy Seizure Prediction\saved_models\seizure_prediction_model.keras"
)

print("Model loaded successfully!")

print("\nMaking predictions...")

probabilities = model.predict(X_test)

# Convert probabilities to classes
y_pred = (probabilities >= 0.5).astype(int).flatten()

print("\nPredicted Class Distribution")
print("Normal:", np.sum(y_pred == 0))
print("Preictal:", np.sum(y_pred == 1))

print("\nActual Class Distribution")
print("Normal:", np.sum(y_test == 0))
print("Preictal:", np.sum(y_test == 1))

print("\nConfusion Matrix")

cm = confusion_matrix(y_test, y_pred)

print(cm)

print("\nClassification Report")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Preictal"],
        zero_division=0
    )
)