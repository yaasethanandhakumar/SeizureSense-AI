import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

BASE_DIR = r"C:\Projects\Epilepsy Seizure Prediction"

DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

# Load test data
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print("Original test shape:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# Same transformation used during training
X_test = np.transpose(X_test, (0, 2, 1))

X_test = X_test.astype("float32")

# Load training data only to reproduce training normalization
X_train = np.load(
    os.path.join(DATA_DIR, "X_train.npy")
)

X_train = np.transpose(X_train, (0, 2, 1))
X_train = X_train.astype("float32")

mean = np.mean(X_train)
std = np.std(X_train)

X_test = (X_test - mean) / (std + 1e-8)

print("\nTest data prepared:")
print("X_test:", X_test.shape)

# Load BEST model
model_path = os.path.join(
    MODEL_DIR,
    "cnn_bilstm_attention_best.keras"
)

model = tf.keras.models.load_model(model_path, safe_mode=False)

print("\nModel loaded successfully.")

# Evaluate
results = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\nTest results:")
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")

# Predictions
probabilities = model.predict(
    X_test,
    verbose=1
)

y_pred = (probabilities >= 0.5).astype(int).ravel()

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n==============================")
print("FINAL TEST PERFORMANCE")
print("==============================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "Preictal"
        ]
    )
)