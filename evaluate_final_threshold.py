import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score
)

# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = r"C:\Projects\Epilepsy Seizure Prediction"

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "cnn_bilstm_attention_corrected_best.keras"
)

X_TEST_PATH = os.path.join(
    PROJECT_ROOT,
    "clean_project",
    "dataset",
    "final",
    "X_test.npy"
)

Y_TEST_PATH = os.path.join(
    PROJECT_ROOT,
    "clean_project",
    "dataset",
    "final",
    "y_test.npy"
)

THRESHOLD_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "selected_validation_threshold.txt"
)

# ============================================================
# 2. TRAINING NORMALIZATION STATISTICS
# ============================================================

TRAINING_MEAN = 2.8031432730557526e-09
TRAINING_STD = 4.079479549545795e-05

# ============================================================
# 3. CORRECT ATTENTION LAYER
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="EEG",
    name="AttentionLayer"
)
class AttentionLayer(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):

        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True
        )

        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True
        )

        self.V = self.add_weight(
            name="attention_vector",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )

        super().build(input_shape)

    def call(self, inputs):

        score = tf.tanh(
            tf.matmul(inputs, self.W) + self.b
        )

        score = tf.matmul(score, self.V)

        weights = tf.nn.softmax(
            score,
            axis=1
        )

        context = inputs * weights

        return tf.reduce_sum(
            context,
            axis=1
        )

    def get_config(self):
        return super().get_config()


# ============================================================
# 4. LOAD VALIDATION-SELECTED THRESHOLD
# ============================================================

if not os.path.exists(THRESHOLD_PATH):
    raise FileNotFoundError(
        f"Threshold file not found:\n{THRESHOLD_PATH}"
    )

with open(THRESHOLD_PATH, "r") as f:
    threshold = float(f.read().strip())

print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

print(f"\nValidation-selected threshold: {threshold:.2f}")

if threshold != 0.88:
    print(
        "\nWARNING: The threshold is not 0.88. "
        "Check selected_validation_threshold.txt."
    )

# ============================================================
# 5. LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# ============================================================
# 6. CHECK TEST DATA
# ============================================================

if X_test.ndim != 3:
    raise ValueError(
        f"Expected X_test to have 3 dimensions, got {X_test.ndim}"
    )

if X_test.shape[1:] != (2560, 23):
    raise ValueError(
        f"Unexpected X_test shape: {X_test.shape}"
    )

if len(X_test) != len(y_test):
    raise ValueError(
        "X_test and y_test contain different numbers of samples."
    )

print("\nTest samples:", len(X_test))

print(
    "Normal samples:",
    int(np.sum(y_test == 0))
)

print(
    "Preictal samples:",
    int(np.sum(y_test == 1))
)

# ============================================================
# 7. NORMALIZE USING TRAINING STATISTICS
# ============================================================

print("\nApplying training normalization...")

X_test = (
    X_test - TRAINING_MEAN
) / TRAINING_STD

print(
    "Normalized mean:",
    float(np.mean(X_test))
)

print(
    "Normalized std:",
    float(np.std(X_test))
)

# ============================================================
# 8. LOAD INTACT SAVED MODEL
# ============================================================

print("\nLoading saved model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "AttentionLayer": AttentionLayer
    },
    compile=False
)

print("Model loaded successfully.")
print("Model parameters:", model.count_params())

# ============================================================
# 9. GENERATE TEST PROBABILITIES
# ============================================================

print("\nGenerating test predictions...")
print("IMPORTANT: No retraining is performed.")

probabilities = model.predict(
    X_test,
    batch_size=16,
    verbose=1
).reshape(-1)

# ============================================================
# 10. APPLY FIXED VALIDATION THRESHOLD
# ============================================================

y_pred = (
    probabilities >= threshold
).astype(int)

# ============================================================
# 11. CALCULATE FINAL METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)

# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
).ravel()

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)

# ============================================================
# 13. PRINT FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL UNBIASED TEST RESULTS")
print("=" * 70)

print(f"\nThreshold:          {threshold:.2f}")

print(f"Accuracy:           {accuracy * 100:.2f}%")
print(f"Precision:          {precision * 100:.2f}%")
print(f"Recall/Sensitivity: {recall * 100:.2f}%")
print(f"F1 Score:           {f1 * 100:.2f}%")
print(f"ROC-AUC:            {roc_auc * 100:.2f}%")
print(f"Specificity:        {specificity * 100:.2f}%")
print(f"Balanced Accuracy:  {balanced_accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")

print("\nError Analysis:")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")

print("\n" + "=" * 70)
print("TEST EVALUATION COMPLETED")
print("=" * 70)

print(
    "\nThe threshold was selected using the validation set "
    "and then kept fixed for this final test evaluation."
)

print(
    "The test set was NOT used to select or tune the threshold."
)

print("=" * 70)