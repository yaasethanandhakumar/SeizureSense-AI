# ============================================================
# SEIZURESENSE AI
# CORRECTED MODEL EVALUATION
# ============================================================

import os
import glob
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
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"C:\Projects\Epilepsy Seizure Prediction"

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "cnn_bilstm_attention_corrected_best.keras"
)

EVALUATION_DIR = os.path.join(
    PROJECT_ROOT,
    "evaluation"
)


# ============================================================
# NORMALIZATION
# ============================================================

TRAINING_MEAN = 2.8031432730557526e-09
TRAINING_STD = 4.079479549545795e-05


# ============================================================
# CORRECT ATTENTION LAYER
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="EEG",
    name="AttentionLayer"
)
class AttentionLayer(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):

        # W: (128, 128)
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True
        )

        # b: (128,)
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True
        )

        # V: (128, 1)
        self.V = self.add_weight(
            name="attention_vector",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )

        super().build(input_shape)

    def call(self, inputs):

        # First attention transformation
        score = tf.tanh(
            tf.matmul(inputs, self.W) + self.b
        )

        # Convert 128-dimensional score to one attention score
        score = tf.matmul(
            score,
            self.V
        )

        # Attention weights
        weights = tf.nn.softmax(
            score,
            axis=1
        )

        # Weighted features
        context = inputs * weights

        # Weighted sum over time
        context = tf.reduce_sum(
            context,
            axis=1
        )

        return context

    def get_config(self):

        config = super().get_config()

        return config


# ============================================================
# FIND DATA FILE
# ============================================================

def find_file(filename):

    search_paths = [
        os.path.join(EVALUATION_DIR, filename),
        os.path.join(PROJECT_ROOT, filename)
    ]

    for path in search_paths:

        if os.path.exists(path):
            return path

    matches = glob.glob(
        os.path.join(
            PROJECT_ROOT,
            "**",
            filename
        ),
        recursive=True
    )

    if matches:
        return matches[0]

    return None


# ============================================================
# LOAD DATA
# ============================================================

print("\n==============================================")
print("             SEIZURESENSE AI")
print("          CORRECTED MODEL EVALUATION")
print("==============================================")

print("\nTensorFlow:", tf.__version__)
print("Keras:", keras.__version__)


print("\n==============================================")
print("             LOADING TEST DATA")
print("==============================================")


X_test_path = find_file("X_test.npy")
y_test_path = find_file("y_test.npy")


if X_test_path is None:
    raise FileNotFoundError(
        "Could not find X_test.npy inside the project."
    )

if y_test_path is None:
    raise FileNotFoundError(
        "Could not find y_test.npy inside the project."
    )


print("X_test:", X_test_path)
print("y_test:", y_test_path)


X_test = np.load(X_test_path)
y_test = np.load(y_test_path)


print("\nOriginal X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# INPUT SHAPE
# ============================================================

print("\n==============================================")
print("             INPUT SHAPE CHECK")
print("==============================================")


# Expected final shape:
# (samples, 2560, 23)

if X_test.ndim != 3:

    raise ValueError(
        f"Expected 3D X_test array, got {X_test.shape}"
    )


# If data is (samples, channels, samples)
if X_test.shape[1] == 23 and X_test.shape[2] == 2560:

    print("Transposing X_test...")

    X_test = np.transpose(
        X_test,
        (0, 2, 1)
    )


elif X_test.shape[1] == 2560 and X_test.shape[2] == 23:

    print("X_test already has correct orientation.")


else:

    raise ValueError(
        f"Unexpected X_test shape: {X_test.shape}"
    )


print("Final X_test shape:", X_test.shape)


# ============================================================
# NORMALIZATION
# ============================================================

print("\n==============================================")
print("             NORMALIZING TEST DATA")
print("==============================================")


print("Training mean:", TRAINING_MEAN)
print("Training std :", TRAINING_STD)


X_test = (
    X_test - TRAINING_MEAN
) / TRAINING_STD


print(
    "Normalized test mean:",
    float(np.mean(X_test))
)

print(
    "Normalized test std:",
    float(np.std(X_test))
)


# ============================================================
# LOAD SAVED MODEL DIRECTLY
# ============================================================

print("\n==============================================")
print("          LOADING SAVED MODEL")
print("==============================================")


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


print("Model path:")
print(MODEL_PATH)


print("\nLoading Keras model directly...")


model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "AttentionLayer": AttentionLayer
    },
    compile=False
)


print("\nModel loaded successfully.")


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\n==============================================")
print("             MODEL INFORMATION")
print("==============================================")


print(
    "Input shape :",
    model.input_shape
)

print(
    "Output shape:",
    model.output_shape
)

print(
    "Parameters  :",
    model.count_params()
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\n==============================================")
print("             GENERATING PREDICTIONS")
print("==============================================")


print("Running model prediction...")


y_prob = model.predict(
    X_test,
    batch_size=16,
    verbose=1
)


y_prob = y_prob.reshape(-1)


print("\nPredictions generated.")

print(
    "Minimum probability:",
    float(np.min(y_prob))
)

print(
    "Maximum probability:",
    float(np.max(y_prob))
)

print(
    "Mean probability:",
    float(np.mean(y_prob))
)


# ============================================================
# ROC-AUC
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    y_prob
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_threshold(
    threshold
):

    y_pred = (
        y_prob >= threshold
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

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()

    return (
        accuracy,
        precision,
        recall,
        f1,
        tn,
        fp,
        fn,
        tp
    )


# ============================================================
# BASELINE THRESHOLD
# ============================================================

print("\n==============================================")
print("          THRESHOLD = 0.50")
print("==============================================")


(
    accuracy,
    precision,
    recall,
    f1,
    tn,
    fp,
    fn,
    tp
) = evaluate_threshold(0.50)


print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC   : {roc_auc * 100:.2f}%"
)

print("\nConfusion Matrix:")

print(
    f"TN = {tn}"
)

print(
    f"FP = {fp}"
)

print(
    f"FN = {fn}"
)

print(
    f"TP = {tp}"
)


# ============================================================
# THRESHOLD COMPARISON
# ============================================================

print("\n==============================================")
print("          THRESHOLD COMPARISON")
print("==============================================")

print(
    "\nThreshold | Accuracy | Precision | Recall | F1 | FP | FN"
)

print(
    "-" * 70
)


thresholds = np.arange(
    0.50,
    0.91,
    0.01
)


results = []


for threshold in thresholds:

    (
        accuracy,
        precision,
        recall,
        f1,
        tn,
        fp,
        fn,
        tp
    ) = evaluate_threshold(
        threshold
    )

    results.append(
        (
            threshold,
            accuracy,
            precision,
            recall,
            f1,
            fp,
            fn
        )
    )

    print(
        f"{threshold:8.2f} | "
        f"{accuracy * 100:8.2f}% | "
        f"{precision * 100:9.2f}% | "
        f"{recall * 100:6.2f}% | "
        f"{f1 * 100:6.2f}% | "
        f"{fp:2d} | "
        f"{fn:2d}"
    )


# ============================================================
# BEST THRESHOLD BY F1
# ============================================================

best_f1 = max(
    results,
    key=lambda x: x[4]
)


print("\n==============================================")
print("          BEST F1 THRESHOLD")
print("==============================================")


print(
    f"Threshold : {best_f1[0]:.2f}"
)

print(
    f"Accuracy  : {best_f1[1] * 100:.2f}%"
)

print(
    f"Precision : {best_f1[2] * 100:.2f}%"
)

print(
    f"Recall    : {best_f1[3] * 100:.2f}%"
)

print(
    f"F1 Score  : {best_f1[4] * 100:.2f}%"
)

print(
    f"FP        : {best_f1[5]}"
)

print(
    f"FN        : {best_f1[6]}"
)


# ============================================================
# BEST THRESHOLD BY PRECISION
# ============================================================

best_precision = max(
    results,
    key=lambda x: x[2]
)


print("\n==============================================")
print("       HIGHEST PRECISION THRESHOLD")
print("==============================================")


print(
    f"Threshold : {best_precision[0]:.2f}"
)

print(
    f"Accuracy  : {best_precision[1] * 100:.2f}%"
)

print(
    f"Precision : {best_precision[2] * 100:.2f}%"
)

print(
    f"Recall    : {best_precision[3] * 100:.2f}%"
)

print(
    f"F1 Score  : {best_precision[4] * 100:.2f}%"
)

print(
    f"FP        : {best_precision[5]}"
)

print(
    f"FN        : {best_precision[6]}"
)


# ============================================================
# FINAL
# ============================================================

print("\n==============================================")
print("             EVALUATION COMPLETE")
print("==============================================")

print(
    "\nThe saved model was evaluated without retraining."
)

print(
    "The .keras model itself was not modified."
)

print(
    "\nIMPORTANT:"
)

print(
    "Threshold selection should ultimately be performed"
)

print(
    "using the validation set, not the test set."
)

print(
    "\nNext step: tune the threshold on X_val/y_val and"
)

print(
    "then evaluate that fixed threshold once on X_test/y_test."
)

print(
    "\n==============================================\n"
)