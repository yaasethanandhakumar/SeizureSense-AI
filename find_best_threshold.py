import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ============================================================
# PATHS
# ============================================================

BASE_PATH = r"C:\Projects\Epilepsy Seizure Prediction"

DATA_PATH = os.path.join(
    BASE_PATH,
    "dataset"
)

MODEL_PATH = os.path.join(
    BASE_PATH,
    "saved_models",
    "cnn_bilstm_attention.keras"
)


# ============================================================
# ATTENTION LAYER
# ============================================================

class Attention(Layer):

    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):

        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )

        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )

        super(Attention, self).build(input_shape)

    def call(self, x):

        score = tf.nn.tanh(
            tf.matmul(x, self.W) + self.b
        )

        attention_weights = tf.nn.softmax(
            score,
            axis=1
        )

        context_vector = attention_weights * x

        context_vector = tf.reduce_sum(
            context_vector,
            axis=1
        )

        return context_vector


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

X_test = np.load(
    os.path.join(DATA_PATH, "X_test.npy")
)

y_test = np.load(
    os.path.join(DATA_PATH, "y_test.npy")
)

if X_test.shape[1:] == (23, 2560):

    X_test = np.transpose(
        X_test,
        (0, 2, 1)
    )

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "Attention": Attention
    },
    compile=False
)

print("Model loaded successfully.")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_probability = model.predict(
    X_test,
    batch_size=16,
    verbose=1
).ravel()


# ============================================================
# ROC-AUC
# ============================================================

auc = roc_auc_score(
    y_test,
    y_probability
)

print("\nROC-AUC:", f"{auc:.4f}")


# ============================================================
# THRESHOLD TESTING
# ============================================================

print("\n==============================================")
print("           THRESHOLD ANALYSIS")
print("==============================================")

print(
    f"{'Threshold':>10}"
    f"{'Accuracy':>12}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1-Score':>12}"
)

print("-" * 60)

best_threshold = 0.5
best_f1 = 0

results = []

for threshold in np.arange(
    0.10,
    0.91,
    0.05
):

    y_pred = (
        y_probability >= threshold
    ).astype(int)

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

    print(
        f"{threshold:10.2f}"
        f"{accuracy:12.4f}"
        f"{precision:12.4f}"
        f"{recall:12.4f}"
        f"{f1:12.4f}"
    )

    results.append(
        (
            threshold,
            accuracy,
            precision,
            recall,
            f1
        )
    )

    if f1 > best_f1:

        best_f1 = f1
        best_threshold = threshold


# ============================================================
# BEST THRESHOLD
# ============================================================

print("\n==============================================")
print("             BEST THRESHOLD")
print("==============================================")

print(
    "Threshold:",
    f"{best_threshold:.2f}"
)

print(
    "F1-Score:",
    f"{best_f1:.4f}"
)


# ============================================================
# FINAL PREDICTIONS
# ============================================================

y_pred_best = (
    y_probability >= best_threshold
).astype(int)


# ============================================================
# FINAL METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred_best
)

precision = precision_score(
    y_test,
    y_pred_best,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred_best,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred_best,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred_best
)


# ============================================================
# DISPLAY FINAL RESULTS
# ============================================================

print("\n==============================================")
print("       RESULTS USING BEST THRESHOLD")
print("==============================================")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1-Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {auc:.4f}"
)

print("\nConfusion Matrix:")

print(cm)

print("\nPrediction Distribution:")

print(
    "Normal   :",
    np.sum(y_pred_best == 0)
)

print(
    "Preictal :",
    np.sum(y_pred_best == 1)
)

print("\nThreshold analysis completed successfully!")