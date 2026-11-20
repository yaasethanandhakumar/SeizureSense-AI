import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ============================================================
# MODEL PREDICTION CHECK
# ============================================================

print("\n==========================================")
print("       CHECKING MODEL PREDICTIONS")
print("==========================================")

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Projects\Epilepsy Seizure Prediction\clean_project"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "cnn_bilstm_attention_v3.keras"
)

X_TEST_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "final",
    "X_test.npy"
)

Y_TEST_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "final",
    "y_test.npy"
)

# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking files...\n")

print("Model path:")
print(MODEL_PATH)

print("\nX_test path:")
print(X_TEST_PATH)

print("\ny_test path:")
print(Y_TEST_PATH)

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Model file not found!")
    exit()

if not os.path.exists(X_TEST_PATH):
    print("\nERROR: X_test.npy not found!")
    exit()

if not os.path.exists(Y_TEST_PATH):
    print("\nERROR: y_test.npy not found!")
    exit()

print("\nModel file  : FOUND")
print("X_test file : FOUND")
print("y_test file : FOUND")

# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n==========================================")
print("          LOADING TEST DATA")
print("==========================================")

X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

print("\nX_test shape :", X_test.shape)
print("y_test shape :", y_test.shape)

print("\nOriginal test data:")
print("Mean :", np.mean(X_test))
print("Std  :", np.std(X_test))
print("Min  :", np.min(X_test))
print("Max  :", np.max(X_test))

# ============================================================
# IMPORTANT:
# NORMALIZATION MUST MATCH TRAINING
# ============================================================

print("\n==========================================")
print("       NORMALIZING TEST DATA")
print("==========================================")

def normalize_eeg(X):

    # Calculate mean across TIME dimension
    mean = np.mean(
        X,
        axis=1,
        keepdims=True
    )

    # Calculate standard deviation across TIME dimension
    std = np.std(
        X,
        axis=1,
        keepdims=True
    )

    # Prevent division by zero
    std = np.maximum(
        std,
        1e-8
    )

    # Normalize
    X = (X - mean) / std

    return X


# Apply EXACT SAME normalization used during training
X_test = normalize_eeg(X_test)

print("\nNormalized test data:")
print("Mean :", np.mean(X_test))
print("Std  :", np.std(X_test))
print("Min  :", np.min(X_test))
print("Max  :", np.max(X_test))

# ============================================================
# CHECK LABEL DISTRIBUTION
# ============================================================

print("\nTest label distribution:")

print(
    "Normal   :",
    np.sum(y_test == 0)
)

print(
    "Preictal :",
    np.sum(y_test == 1)
)

# ============================================================
# LOAD MODEL
# ============================================================

print("\n==========================================")
print("          LOADING TRAINED MODEL")
print("==========================================")

# Register custom Attention layer
@tf.keras.utils.register_keras_serializable()
class Attention(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.score_dense = tf.keras.layers.Dense(
            1,
            activation="tanh"
        )

    def call(self, inputs):

        # Attention scores
        scores = self.score_dense(inputs)

        # Attention weights
        weights = tf.nn.softmax(
            scores,
            axis=1
        )

        # Weighted sequence
        weighted = inputs * weights

        # Sum over time
        output = tf.reduce_sum(
            weighted,
            axis=1
        )

        return output

    def get_config(self):

        config = super().get_config()

        return config


try:

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "Attention": Attention
        },
        compile=False,
        safe_mode=False
    )

    print("\nModel loaded successfully!")

except Exception as e:

    print("\nERROR LOADING MODEL:")
    print(e)
    exit()

# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n==========================================")
print("             MODEL SUMMARY")
print("==========================================\n")

model.summary()

# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\n==========================================")
print("        GENERATING PREDICTIONS")
print("==========================================")

predictions = model.predict(
    X_test,
    batch_size=32,
    verbose=1
)

predictions = predictions.reshape(-1)

# ============================================================
# PROBABILITY CHECK
# ============================================================

print("\n==========================================")
print("          PROBABILITY CHECK")
print("==========================================")

print(
    "Minimum probability :",
    np.min(predictions)
)

print(
    "Maximum probability :",
    np.max(predictions)
)

print(
    "Mean probability    :",
    np.mean(predictions)
)

print(
    "Std probability     :",
    np.std(predictions)
)

print("\nFirst 20 probabilities:")

for i in range(min(20, len(predictions))):

    print(
        f"Sample {i+1:02d} : "
        f"{predictions[i]:.6f}"
    )

# ============================================================
# CLASSIFICATION THRESHOLD
# ============================================================

THRESHOLD = 0.5

predicted_classes = (
    predictions >= THRESHOLD
).astype(int)

# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print("\n==========================================")
print("       PREDICTION DISTRIBUTION")
print("==========================================")

print(
    "\nActual Normal    :",
    np.sum(y_test == 0)
)

print(
    "Actual Preictal  :",
    np.sum(y_test == 1)
)

print(
    "\nPredicted Normal  :",
    np.sum(predicted_classes == 0)
)

print(
    "Predicted Preictal:",
    np.sum(predicted_classes == 1)
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predicted_classes
)

print("\n==========================================")
print("          CONFUSION MATRIX")
print("==========================================\n")

print(cm)

# ============================================================
# METRICS
# ============================================================

accuracy = np.mean(
    predicted_classes == y_test
)

precision = precision_score(
    y_test,
    predicted_classes,
    zero_division=0
)

recall = recall_score(
    y_test,
    predicted_classes,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predicted_classes,
    zero_division=0
)

# AUC needs probabilities, not class predictions
try:

    auc = roc_auc_score(
        y_test,
        predictions
    )

except Exception:

    auc = 0.0

# ============================================================
# FINAL METRICS
# ============================================================

print("\n==========================================")
print("            FINAL METRICS")
print("==========================================")

print(
    "Accuracy  :",
    round(accuracy * 100, 2),
    "%"
)

print(
    "Precision :",
    round(precision * 100, 2),
    "%"
)

print(
    "Recall    :",
    round(recall * 100, 2),
    "%"
)

print(
    "F1 Score  :",
    round(f1 * 100, 2),
    "%"
)

print(
    "AUC       :",
    round(auc, 4)
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n==========================================")
print("        CLASSIFICATION REPORT")
print("==========================================\n")

print(
    classification_report(
        y_test,
        predicted_classes,
        target_names=[
            "Normal",
            "Preictal"
        ],
        zero_division=0
    )
)

# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n==========================================")
print("          SAMPLE PREDICTIONS")
print("==========================================\n")

for i in range(
    min(20, len(y_test))
):

    actual = (
        "Preictal"
        if y_test[i] == 1
        else "Normal"
    )

    predicted = (
        "Preictal"
        if predicted_classes[i] == 1
        else "Normal"
    )

    probability = predictions[i]

    print(
        f"Sample {i+1:02d} | "
        f"Actual: {actual:8s} | "
        f"Predicted: {predicted:8s} | "
        f"Probability: {probability:.4f}"
    )

# ============================================================
# COMPLETED
# ============================================================

print("\n==========================================")
print("       PREDICTION CHECK COMPLETED")
print("==========================================")