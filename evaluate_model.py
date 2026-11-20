import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = r"C:\Projects\Epilepsy Seizure Prediction\dataset"

MODEL_PATH = (
    r"C:\Projects\Epilepsy Seizure Prediction"
    r"\saved_models\cnn_bilstm_attention.keras"
)


# ============================================================
# 2. CUSTOM ATTENTION LAYER
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
# 3. LOAD TEST DATA
# ============================================================

print("\n========================================")
print("        LOADING TEST DATA")
print("========================================")

X_test = np.load(
    DATA_PATH + r"\X_test.npy"
)

y_test = np.load(
    DATA_PATH + r"\y_test.npy"
)

# Original data:
# (samples, 23, 2560)
#
# Model expects:
# (samples, 2560, 23)

if X_test.shape[1:] == (23, 2560):

    X_test = np.transpose(
        X_test,
        (0, 2, 1)
    )

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# 4. LOAD TRAINED MODEL
# ============================================================

print("\n========================================")
print("        LOADING TRAINED MODEL")
print("========================================")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "Attention": Attention
    },
    compile=False
)

print("Model loaded successfully!")


# ============================================================
# 5. GENERATE PREDICTIONS
# ============================================================

print("\n========================================")
print("        GENERATING PREDICTIONS")
print("========================================")

y_probability = model.predict(
    X_test,
    batch_size=16,
    verbose=1
)

# Convert (467, 1) → (467,)
y_probability = y_probability.ravel()

# Convert probability to class
# 0 = Normal
# 1 = Preictal

y_pred = (
    y_probability >= 0.5
).astype(int)


# ============================================================
# 6. CALCULATE PERFORMANCE METRICS
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

auc = roc_auc_score(
    y_test,
    y_probability
)


# ============================================================
# 7. DISPLAY FINAL RESULTS
# ============================================================

print("\n========================================")
print("          FINAL TEST RESULTS")
print("========================================")

print(
    f"Accuracy  : {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
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


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n========================================")
print("          CONFUSION MATRIX")
print("========================================")

print(cm)

print("\nFormat:")
print("[[True Negative   False Positive]")
print(" [False Negative  True Positive]]")


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("       CLASSIFICATION REPORT")
print("========================================")

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Normal",
        "Preictal"
    ],
    zero_division=0
)

print(report)


# ============================================================
# 10. PREDICTION DISTRIBUTION
# ============================================================

normal_predictions = np.sum(
    y_pred == 0
)

preictal_predictions = np.sum(
    y_pred == 1
)

print("\n========================================")
print("       PREDICTION DISTRIBUTION")
print("========================================")

print(
    "Predicted Normal   :",
    normal_predictions
)

print(
    "Predicted Preictal :",
    preictal_predictions
)


# ============================================================
# 11. ACTUAL CLASS DISTRIBUTION
# ============================================================

actual_normal = np.sum(
    y_test == 0
)

actual_preictal = np.sum(
    y_test == 1
)

print("\n========================================")
print("          ACTUAL DISTRIBUTION")
print("========================================")

print(
    "Actual Normal   :",
    actual_normal
)

print(
    "Actual Preictal :",
    actual_preictal
)


# ============================================================
# 12. COMPLETION
# ============================================================

print("\n========================================")
print("     EVALUATION COMPLETED SUCCESSFULLY")
print("========================================")