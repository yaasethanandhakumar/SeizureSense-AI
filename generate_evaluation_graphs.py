# ============================================================
# FINAL MODEL EVALUATION GRAPHS
# Epilepsy Seizure Prediction Using Multi-Channel EEG
# ============================================================

import os
import json
import numpy as np
import tensorflow as tf
import keras

from keras.layers import Layer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"C:\Projects\Epilepsy Seizure Prediction"

X_TEST_PATH = os.path.join(
    PROJECT_DIR,
    "dataset",
    "X_test.npy"
)

Y_TEST_PATH = os.path.join(
    PROJECT_DIR,
    "dataset",
    "y_test.npy"
)

NORMALIZATION_PATH = os.path.join(
    PROJECT_DIR,
    "saved_models",
    "normalization_stats.json"
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "saved_models",
    "cnn_bilstm_attention_corrected.keras"
)

HISTORY_PATH = os.path.join(
    PROJECT_DIR,
    "saved_models",
    "cnn_bilstm_attention_corrected_history.npy"
)

GRAPH_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "graphs"
)

os.makedirs(GRAPH_DIR, exist_ok=True)


# ============================================================
# CUSTOM ATTENTION LAYER
# ============================================================

@keras.saving.register_keras_serializable(
    package="EEG",
    name="AttentionLayer"
)
class AttentionLayer(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):

        self.attention_weight = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )

        self.attention_bias = self.add_weight(
            name="attention_bias",
            shape=(1,),
            initializer="zeros",
            trainable=True
        )

        super().build(input_shape)

    def call(self, inputs):

        score = tf.tanh(
            tf.matmul(
                inputs,
                self.attention_weight
            ) + self.attention_bias
        )

        weights = tf.nn.softmax(
            score,
            axis=1
        )

        context = inputs * weights

        context = tf.reduce_sum(
            context,
            axis=1
        )

        return context

    def get_config(self):

        config = super().get_config()

        return config

# ============================================================
# START
# ============================================================

print("=" * 60)
print("       FINAL MODEL EVALUATION GRAPHS")
print("=" * 60)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n==============================================")
print("             LOADING TEST DATA")
print("==============================================")

print("\nX_test path:")
print(X_TEST_PATH)

print("\ny_test path:")
print(Y_TEST_PATH)

if not os.path.exists(X_TEST_PATH):
    raise FileNotFoundError(
        f"X_test not found:\n{X_TEST_PATH}"
    )

if not os.path.exists(Y_TEST_PATH):
    raise FileNotFoundError(
        f"y_test not found:\n{Y_TEST_PATH}"
    )

X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

print("\nOriginal X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# INPUT SHAPE CHECK
# ============================================================

print("\n==============================================")
print("             INPUT SHAPE CHECK")
print("==============================================")

if X_test.ndim != 3:
    raise ValueError(
        f"Expected 3D X_test array. Got shape: {X_test.shape}"
    )

# Expected:
# (samples, channels, time)
# -> (samples, time, channels)

if X_test.shape[1] == 23 and X_test.shape[2] == 2560:

    print(
        "Transposing:",
        X_test.shape,
        "-> (samples, 2560, 23)"
    )

    X_test = np.transpose(
        X_test,
        (0, 2, 1)
    )

elif X_test.shape[1] == 2560 and X_test.shape[2] == 23:

    print("X_test is already in correct format.")

else:

    raise ValueError(
        f"Unexpected X_test shape: {X_test.shape}"
    )

print("Final X_test shape:", X_test.shape)


# ============================================================
# LOAD NORMALIZATION STATISTICS
# ============================================================

print("\n==============================================")
print("             NORMALIZATION")
print("==============================================")

if not os.path.exists(NORMALIZATION_PATH):

    raise FileNotFoundError(
        f"Normalization file not found:\n"
        f"{NORMALIZATION_PATH}"
    )

with open(
    NORMALIZATION_PATH,
    "r"
) as f:

    stats = json.load(f)


# Your JSON contains:
# training_mean
# training_std

if "training_mean" in stats:

    training_mean = stats["training_mean"]

elif "mean" in stats:

    training_mean = stats["mean"]

else:

    raise KeyError(
        "Could not find training mean in normalization_stats.json"
    )


if "training_std" in stats:

    training_std = stats["training_std"]

elif "std" in stats:

    training_std = stats["std"]

else:

    raise KeyError(
        "Could not find training std in normalization_stats.json"
    )


print("Training mean:", training_mean)
print("Training std :", training_std)


# Avoid division by zero

if training_std == 0:

    raise ValueError(
        "Training standard deviation is zero."
    )


# ============================================================
# NORMALIZE TEST DATA
# ============================================================

X_test_normalized = (
    X_test - training_mean
) / training_std

print(
    "Normalized test mean:",
    np.mean(X_test_normalized)
)

print(
    "Normalized test std :",
    np.std(X_test_normalized)
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n==============================================")
print("             LOADING FINAL MODEL")
print("==============================================")

print("\nModel:")
print(MODEL_PATH)

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


try:

    model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "AttentionLayer": AttentionLayer
    },
    compile=False
)

    print("\nModel loaded successfully.")

except Exception as e:

    print("\nMODEL LOADING FAILED")
    print("-" * 50)
    print(e)

    raise


print("\nInput shape :", model.input_shape)
print("Output shape:", model.output_shape)
print(
    "Parameters  :",
    model.count_params()
)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\n==============================================")
print("             GENERATING PREDICTIONS")
print("==============================================")

probabilities = model.predict(
    X_test_normalized,
    batch_size=16,
    verbose=1
)


# Convert shape:
# (467, 1) -> (467,)

probabilities = np.asarray(
    probabilities
).reshape(-1)


print("\nProbability statistics:")
print("Minimum :", np.min(probabilities))
print("Maximum :", np.max(probabilities))
print("Mean    :", np.mean(probabilities))
print("Median  :", np.median(probabilities))
print("Std     :", np.std(probabilities))


# ============================================================
# THRESHOLD
# ============================================================

THRESHOLD = 0.50

y_pred = (
    probabilities >= THRESHOLD
).astype(int)


# ============================================================
# CALCULATE METRICS
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


print("\n==============================================")
print("             FINAL RESULTS")
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
    f"ROC-AUC   : {roc_auc:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# GRAPH 1 — CONFUSION MATRIX
# ============================================================

print("\nGenerating confusion matrix graph...")

fig, ax = plt.subplots(
    figsize=(7, 6)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Normal",
        "Preictal"
    ]
)

display.plot(
    ax=ax,
    values_format="d"
)

ax.set_title(
    "Confusion Matrix - CNN-BiLSTM-Attention"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRAPH 2 — ROC CURVE
# ============================================================

print("Generating ROC curve...")

fpr, tpr, _ = roc_curve(
    y_test,
    probabilities
)

fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.4f}"
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

ax.set_xlabel(
    "False Positive Rate"
)

ax.set_ylabel(
    "True Positive Rate"
)

ax.set_title(
    "ROC Curve - CNN-BiLSTM-Attention"
)

ax.legend()

ax.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "roc_curve.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRAPH 3 — PRECISION-RECALL CURVE
# ============================================================

print("Generating Precision-Recall curve...")

pr_precision, pr_recall, _ = precision_recall_curve(
    y_test,
    probabilities
)

fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.plot(
    pr_recall,
    pr_precision
)

ax.set_xlabel(
    "Recall"
)

ax.set_ylabel(
    "Precision"
)

ax.set_title(
    "Precision-Recall Curve - CNN-BiLSTM-Attention"
)

ax.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "precision_recall_curve.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRAPH 4 — PREDICTION DISTRIBUTION
# ============================================================

print("Generating prediction distribution...")

fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.hist(
    probabilities[y_test == 0],
    bins=30,
    alpha=0.7,
    label="Normal"
)

ax.hist(
    probabilities[y_test == 1],
    bins=30,
    alpha=0.7,
    label="Preictal"
)

ax.axvline(
    THRESHOLD,
    linestyle="--",
    label=f"Threshold = {THRESHOLD}"
)

ax.set_xlabel(
    "Predicted Preictal Probability"
)

ax.set_ylabel(
    "Number of Samples"
)

ax.set_title(
    "Prediction Probability Distribution"
)

ax.legend()

ax.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "prediction_distribution.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRAPH 5 — ACTUAL VS PREDICTED DISTRIBUTION
# ============================================================

print("Generating class distribution graph...")

actual_normal = np.sum(
    y_test == 0
)

actual_preictal = np.sum(
    y_test == 1
)

predicted_normal = np.sum(
    y_pred == 0
)

predicted_preictal = np.sum(
    y_pred == 1
)

labels = [
    "Normal",
    "Preictal"
]

actual_values = [
    actual_normal,
    actual_preictal
]

predicted_values = [
    predicted_normal,
    predicted_preictal
]

x = np.arange(
    len(labels)
)

width = 0.35

fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.bar(
    x - width / 2,
    actual_values,
    width,
    label="Actual"
)

ax.bar(
    x + width / 2,
    predicted_values,
    width,
    label="Predicted"
)

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.set_ylabel(
    "Number of Samples"
)

ax.set_title(
    "Actual vs Predicted Class Distribution"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "actual_vs_predicted.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# TRAINING HISTORY GRAPHS
# ============================================================

print("\n==============================================")
print("          TRAINING HISTORY GRAPHS")
print("==============================================")


if os.path.exists(HISTORY_PATH):

    print(
        "Loading:",
        HISTORY_PATH
    )

    history = np.load(
        HISTORY_PATH,
        allow_pickle=True
    )

    # History may be saved as a dictionary

    if isinstance(history, np.ndarray):

        try:

            history = history.item()

        except Exception:

            history = None


    if isinstance(history, dict):

        print(
            "History keys:",
            list(history.keys())
        )


        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        if "accuracy" in history:

            fig, ax = plt.subplots(
                figsize=(9, 6)
            )

            ax.plot(
                history["accuracy"],
                label="Training Accuracy"
            )

            if "val_accuracy" in history:

                ax.plot(
                    history["val_accuracy"],
                    label="Validation Accuracy"
                )

            ax.set_xlabel(
                "Epoch"
            )

            ax.set_ylabel(
                "Accuracy"
            )

            ax.set_title(
                "Training and Validation Accuracy"
            )

            ax.legend()

            ax.grid(
                True,
                alpha=0.3
            )

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    GRAPH_DIR,
                    "accuracy_curve_final.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()


        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        if "loss" in history:

            fig, ax = plt.subplots(
                figsize=(9, 6)
            )

            ax.plot(
                history["loss"],
                label="Training Loss"
            )

            if "val_loss" in history:

                ax.plot(
                    history["val_loss"],
                    label="Validation Loss"
                )

            ax.set_xlabel(
                "Epoch"
            )

            ax.set_ylabel(
                "Loss"
            )

            ax.set_title(
                "Training and Validation Loss"
            )

            ax.legend()

            ax.grid(
                True,
                alpha=0.3
            )

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    GRAPH_DIR,
                    "loss_curve_final.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()


        # ----------------------------------------------------
        # AUC
        # ----------------------------------------------------

        if "auc" in history:

            fig, ax = plt.subplots(
                figsize=(9, 6)
            )

            ax.plot(
                history["auc"],
                label="Training AUC"
            )

            if "val_auc" in history:

                ax.plot(
                    history["val_auc"],
                    label="Validation AUC"
                )

            ax.set_xlabel(
                "Epoch"
            )

            ax.set_ylabel(
                "AUC"
            )

            ax.set_title(
                "Training and Validation AUC"
            )

            ax.legend()

            ax.grid(
                True,
                alpha=0.3
            )

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    GRAPH_DIR,
                    "auc_curve_final.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()


        # ----------------------------------------------------
        # Precision
        # ----------------------------------------------------

        if "precision" in history:

            fig, ax = plt.subplots(
                figsize=(9, 6)
            )

            ax.plot(
                history["precision"],
                label="Training Precision"
            )

            if "val_precision" in history:

                ax.plot(
                    history["val_precision"],
                    label="Validation Precision"
                )

            ax.set_xlabel(
                "Epoch"
            )

            ax.set_ylabel(
                "Precision"
            )

            ax.set_title(
                "Training and Validation Precision"
            )

            ax.legend()

            ax.grid(
                True,
                alpha=0.3
            )

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    GRAPH_DIR,
                    "precision_curve_final.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()


        # ----------------------------------------------------
        # Recall
        # ----------------------------------------------------

        if "recall" in history:

            fig, ax = plt.subplots(
                figsize=(9, 6)
            )

            ax.plot(
                history["recall"],
                label="Training Recall"
            )

            if "val_recall" in history:

                ax.plot(
                    history["val_recall"],
                    label="Validation Recall"
                )

            ax.set_xlabel(
                "Epoch"
            )

            ax.set_ylabel(
                "Recall"
            )

            ax.set_title(
                "Training and Validation Recall"
            )

            ax.legend()

            ax.grid(
                True,
                alpha=0.3
            )

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    GRAPH_DIR,
                    "recall_curve_final.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()


else:

    print(
        "\nTraining history file not found."
    )

    print(
        HISTORY_PATH
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "model": os.path.basename(
        MODEL_PATH
    ),

    "test_samples": int(
        len(y_test)
    ),

    "normal_samples": int(
        np.sum(y_test == 0)
    ),

    "preictal_samples": int(
        np.sum(y_test == 1)
    ),

    "threshold": THRESHOLD,

    "accuracy": float(
        accuracy
    ),

    "precision": float(
        precision
    ),

    "recall": float(
        recall
    ),

    "f1_score": float(
        f1
    ),

    "roc_auc": float(
        roc_auc
    ),

    "confusion_matrix": cm.tolist(),

    "predicted_normal": int(
        predicted_normal
    ),

    "predicted_preictal": int(
        predicted_preictal
    )
}


RESULTS_PATH = os.path.join(
    GRAPH_DIR,
    "evaluation_results.json"
)


with open(
    RESULTS_PATH,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("       GRAPH GENERATION COMPLETED")
print("=" * 60)

print("\nGraphs saved to:")

print(
    GRAPH_DIR
)

print("\nGenerated files:")

for filename in sorted(
    os.listdir(GRAPH_DIR)
):

    if filename.endswith(".png"):

        print(
            " -",
            filename
        )

print("\nFinal metrics:")

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
    f"ROC-AUC   : {roc_auc:.4f}"
)

print("\n" + "=" * 60)

