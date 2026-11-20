# ============================================================
# SEIZURESENSE AI
# VALIDATION THRESHOLD TUNING
# ============================================================

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
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"C:\Projects\Epilepsy Seizure Prediction"

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "clean_project",
    "dataset",
    "final"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "cnn_bilstm_attention_corrected_best.keras"
)


# ============================================================
# NORMALIZATION
# ============================================================

TRAINING_MEAN = 2.8031432730557526e-09
TRAINING_STD = 4.079479549545795e-05


# ============================================================
# ATTENTION LAYER
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

        score = tf.matmul(
            score,
            self.V
        )

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
# START
# ============================================================

print("\n==============================================")
print("       VALIDATION THRESHOLD TUNING")
print("==============================================")

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\n==============================================")
print("          LOADING VALIDATION DATA")
print("==============================================")


X_val_path = os.path.join(
    DATA_DIR,
    "X_val.npy"
)

y_val_path = os.path.join(
    DATA_DIR,
    "y_val.npy"
)


X_val = np.load(
    X_val_path
)

y_val = np.load(
    y_val_path
)


print(
    "Original X_val:",
    X_val.shape
)

print(
    "y_val:",
    y_val.shape
)


# ============================================================
# INPUT SHAPE
# ============================================================

print("\n==============================================")
print("          INPUT SHAPE CHECK")
print("==============================================")


if X_val.shape[1] == 23 and X_val.shape[2] == 2560:

    print("Transposing validation data...")

    X_val = np.transpose(
        X_val,
        (0, 2, 1)
    )


elif X_val.shape[1] == 2560 and X_val.shape[2] == 23:

    print(
        "Validation data already has correct orientation."
    )


else:

    raise ValueError(
        f"Unexpected X_val shape: {X_val.shape}"
    )


print(
    "Final X_val:",
    X_val.shape
)


# ============================================================
# NORMALIZATION
# ============================================================

print("\n==============================================")
print("          NORMALIZING VALIDATION DATA")
print("==============================================")


X_val = (
    X_val - TRAINING_MEAN
) / TRAINING_STD


print(
    "Training mean:",
    TRAINING_MEAN
)

print(
    "Training std:",
    TRAINING_STD
)

print(
    "Normalized validation mean:",
    float(np.mean(X_val))
)

print(
    "Normalized validation std:",
    float(np.std(X_val))
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

normal_count = np.sum(
    y_val == 0
)

preictal_count = np.sum(
    y_val == 1
)


print("\n==============================================")
print("          VALIDATION LABELS")
print("==============================================")


print(
    "Normal   :",
    int(normal_count)
)

print(
    "Preictal :",
    int(preictal_count)
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n==============================================")
print("          LOADING SAVED MODEL")
print("==============================================")


model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "AttentionLayer": AttentionLayer
    },
    compile=False
)


print(
    "Model loaded successfully."
)

print(
    "Parameters:",
    model.count_params()
)


# ============================================================
# PREDICT VALIDATION SET
# ============================================================

print("\n==============================================")
print("       GENERATING VALIDATION PREDICTIONS")
print("==============================================")


y_prob = model.predict(
    X_val,
    batch_size=16,
    verbose=1
)


y_prob = y_prob.reshape(-1)


print("\nPrediction statistics:")

print(
    "Minimum:",
    float(np.min(y_prob))
)

print(
    "Maximum:",
    float(np.max(y_prob))
)

print(
    "Mean:",
    float(np.mean(y_prob))
)


# ============================================================
# ROC-AUC
# ============================================================

roc_auc = roc_auc_score(
    y_val,
    y_prob
)


print(
    "\nValidation ROC-AUC:",
    f"{roc_auc * 100:.2f}%"
)


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

def evaluate_threshold(threshold):

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_val,
        y_pred
    )

    precision = precision_score(
        y_val,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_pred,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        y_pred
    ).ravel()

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
# THRESHOLD SWEEP
# ============================================================

print("\n==============================================")
print("          VALIDATION THRESHOLD SWEEP")
print("==============================================")

print(
    "\nThreshold | Accuracy | Precision | Recall | F1 | FP | FN"
)

print(
    "-" * 70
)


results = []


thresholds = np.arange(
    0.50,
    0.96,
    0.01
)


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
            tn,
            fp,
            fn,
            tp
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
# BEST F1
# ============================================================

best_f1 = max(
    results,
    key=lambda x: x[4]
)


print("\n==============================================")
print("       SELECTED VALIDATION THRESHOLD")
print("==============================================")


print(
    "Selection criterion: Highest validation F1"
)

print(
    f"\nThreshold : {best_f1[0]:.2f}"
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
    f"TN        : {best_f1[5]}"
)

print(
    f"FP        : {best_f1[6]}"
)

print(
    f"FN        : {best_f1[7]}"
)

print(
    f"TP        : {best_f1[8]}"
)


# ============================================================
# HIGH-PRECISION OPTIONS
# ============================================================

print("\n==============================================")
print("        HIGH-PRECISION OPTIONS")
print("==============================================")


precision_candidates = [
    r for r in results
    if r[2] >= 0.80
]


if precision_candidates:

    print(
        "\nThreshold | Precision | Recall | F1 | FP | FN"
    )

    print(
        "-" * 60
    )

    for r in precision_candidates:

        print(
            f"{r[0]:8.2f} | "
            f"{r[2] * 100:9.2f}% | "
            f"{r[3] * 100:6.2f}% | "
            f"{r[4] * 100:6.2f}% | "
            f"{r[6]:2d} | "
            f"{r[7]:2d}"
        )

else:

    print(
        "No threshold reached 80% validation precision."
    )


# ============================================================
# SAVE SELECTED THRESHOLD
# ============================================================

threshold_file = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "selected_validation_threshold.txt"
)


os.makedirs(
    os.path.dirname(threshold_file),
    exist_ok=True
)


with open(
    threshold_file,
    "w"
) as f:

    f.write(
        f"{best_f1[0]:.2f}"
    )


print("\n==============================================")
print("          THRESHOLD SAVED")
print("==============================================")


print(
    "Saved to:"
)

print(
    threshold_file
)


print("\n==============================================")
print("        VALIDATION TUNING COMPLETE")
print("==============================================")

print(
    "\nIMPORTANT:"
)

print(
    "The test set was NOT used to select this threshold."
)

print(
    "The selected threshold will now be fixed."
)

print(
    "Next step: evaluate the fixed threshold on X_test."
)

print(
    "\n==============================================\n"
)