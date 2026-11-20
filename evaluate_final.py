import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
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
    "cnn_bilstm_attention_corrected.keras"
)

STATS_PATH = os.path.join(
    BASE_PATH,
    "saved_models",
    "normalization_stats.json"
)

OUTPUT_PATH = os.path.join(
    BASE_PATH,
    "outputs",
    "final_evaluation.txt"
)


# ============================================================
# CUSTOM ATTENTION LAYER
# ============================================================

@tf.keras.utils.register_keras_serializable(
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

print("\n==============================================")
print("       FINAL MODEL EVALUATION")
print("==============================================")

print("TensorFlow:", tf.__version__)
print("Keras:", tf.keras.__version__)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n==============================================")
print("             LOADING TEST DATA")
print("==============================================")

X_test = np.load(
    os.path.join(
        DATA_PATH,
        "X_test.npy"
    )
)

y_test = np.load(
    os.path.join(
        DATA_PATH,
        "y_test.npy"
    )
)

print("Original X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# INPUT SHAPE CHECK
# ============================================================

print("\n==============================================")
print("             INPUT SHAPE CHECK")
print("==============================================")

if X_test.shape[1:] == (23, 2560):

    print("Transposing X_test...")

    X_test = np.transpose(
        X_test,
        (0, 2, 1)
    )

print("Final X_test:", X_test.shape)


# ============================================================
# LOAD NORMALIZATION STATISTICS
# ============================================================

print("\n==============================================")
print("          NORMALIZING TEST DATA")
print("==============================================")

with open(
    STATS_PATH,
    "r"
) as f:

    stats = json.load(f)


training_mean = stats["training_mean"]
training_std = stats["training_std"]


print(
    "Training mean:",
    training_mean
)

print(
    "Training std :",
    training_std
)


# ============================================================
# NORMALIZATION
# ============================================================

X_test = (
    X_test - training_mean
) / (
    training_std + 1e-8
)


print(
    "Normalized test mean:",
    np.mean(X_test)
)

print(
    "Normalized test std:",
    np.std(X_test)
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\n==============================================")
print("           LOADING TRAINED MODEL")
print("==============================================")

print(
    "Model:",
    MODEL_PATH
)


try:

    model = load_model(
        MODEL_PATH,
        custom_objects={
            "AttentionLayer": AttentionLayer,
            "EEG>AttentionLayer": AttentionLayer
        },
        compile=False
    )

    print(
        "Model loaded successfully."
    )

except Exception as e:

    print("\nMODEL LOADING FAILED")
    print("----------------------------------------------")
    print(e)

    raise SystemExit


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\n==============================================")
print("             MODEL INFORMATION")
print("==============================================")

print(
    "Input shape:",
    model.input_shape
)

print(
    "Output shape:",
    model.output_shape
)

print(
    "Total parameters:",
    model.count_params()
)


# ============================================================
# GENERATE PROBABILITIES
# ============================================================

print("\n==============================================")
print("          GENERATING PREDICTIONS")
print("==============================================")

y_probability = model.predict(
    X_test,
    batch_size=16,
    verbose=1
)

y_probability = y_probability.ravel()


print(
    "\nProbability range:"
)

print(
    "Minimum:",
    np.min(y_probability)
)

print(
    "Maximum:",
    np.max(y_probability)

)

print(
    "Mean:",
    np.mean(y_probability)
)


# ============================================================
# ROC-AUC
# ============================================================

auc = roc_auc_score(
    y_test,
    y_probability
)


print(
    "\nROC-AUC:",
    f"{auc:.4f}"
)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n==============================================")
print("             THRESHOLD ANALYSIS")
print("==============================================")


thresholds = np.arange(
    0.68,
    0.821,
    0.01
)


print(
    "\nThreshold | Accuracy | Precision | Recall | F1 | FP | TN"
)

print(
    "-" * 75
)


threshold_results = []


for threshold in thresholds:

    y_pred_temp = (
        y_probability >= threshold
    ).astype(int)


    accuracy_temp = accuracy_score(
        y_test,
        y_pred_temp
    )


    precision_temp = precision_score(
        y_test,
        y_pred_temp,
        zero_division=0
    )


    recall_temp = recall_score(
        y_test,
        y_pred_temp,
        zero_division=0
    )


    f1_temp = f1_score(
        y_test,
        y_pred_temp,
        zero_division=0
    )


    cm_temp = confusion_matrix(
        y_test,
        y_pred_temp
    )


    tn_temp, fp_temp, fn_temp, tp_temp = (
        cm_temp.ravel()
    )


    threshold_results.append({
        "threshold": threshold,
        "accuracy": accuracy_temp,
        "precision": precision_temp,
        "recall": recall_temp,
        "f1": f1_temp,
        "tn": tn_temp,
        "fp": fp_temp,
        "fn": fn_temp,
        "tp": tp_temp
    })


    print(
        f"{threshold:8.2f} | "
        f"{accuracy_temp:8.4f} | "
        f"{precision_temp:9.4f} | "
        f"{recall_temp:6.4f} | "
        f"{f1_temp:6.4f} | "
        f"{fp_temp:2d} | "
        f"{tn_temp:3d}"
    )


# ============================================================
# BEST THRESHOLDS FOR DIFFERENT METRICS
# ============================================================

best_accuracy = max(
    threshold_results,
    key=lambda x: x["accuracy"]
)

best_precision = max(
    threshold_results,
    key=lambda x: x["precision"]
)

best_f1 = max(
    threshold_results,
    key=lambda x: x["f1"]
)


print("\n==============================================")
print("          THRESHOLD SUMMARY")
print("==============================================")


print(
    "\nBest Accuracy:"
)

print(
    f"Threshold = {best_accuracy['threshold']:.2f}"
)

print(
    f"Accuracy  = {best_accuracy['accuracy']:.4f}"
)

print(
    f"Precision = {best_accuracy['precision']:.4f}"
)

print(
    f"Recall    = {best_accuracy['recall']:.4f}"
)


print(
    "\nBest Precision:"
)

print(
    f"Threshold = {best_precision['threshold']:.2f}"
)

print(
    f"Accuracy  = {best_precision['accuracy']:.4f}"
)

print(
    f"Precision = {best_precision['precision']:.4f}"
)

print(
    f"Recall    = {best_precision['recall']:.4f}"
)


print(
    "\nBest F1-Score:"
)

print(
    f"Threshold = {best_f1['threshold']:.2f}"
)

print(
    f"Accuracy  = {best_f1['accuracy']:.4f}"
)

print(
    f"Precision = {best_f1['precision']:.4f}"
)

print(
    f"Recall    = {best_f1['recall']:.4f}"
)


# ============================================================
# SELECT CURRENT THRESHOLD
# ============================================================

# Keep 0.50 for now.
# We will change this after reviewing the results.

threshold = 0.50


y_pred = (
    y_probability >= threshold
).astype(int)


print("\n==============================================")
print("           SELECTED THRESHOLD")
print("==============================================")


print(
    "Selected threshold:",
    threshold
)


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print("\n==============================================")
print("        PREDICTION DISTRIBUTION")
print("==============================================")


print(
    "Predicted Normal:",
    np.sum(y_pred == 0)
)


print(
    "Predicted Preictal:",
    np.sum(y_pred == 1)
)


# ============================================================
# FINAL METRICS
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


# ============================================================
# FINAL CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


tn, fp, fn, tp = cm.ravel()


# ============================================================
# SPECIFICITY
# ============================================================

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n==============================================")
print("             FINAL TEST RESULTS")
print("==============================================")


print(
    f"Accuracy     : {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)


print(
    f"Precision    : {precision:.4f} "
    f"({precision * 100:.2f}%)"
)


print(
    f"Recall       : {recall:.4f} "
    f"({recall * 100:.2f}%)"
)


print(
    f"F1-Score     : {f1:.4f} "
    f"({f1 * 100:.2f}%)"
)


print(
    f"ROC-AUC      : {auc:.4f} "
    f"({auc * 100:.2f}%)"
)


print(
    f"Specificity  : {specificity:.4f} "
    f"({specificity * 100:.2f}%)"
)


print(
    f"False Positives: {fp}"
)


print(
    f"False Negatives: {fn}"
)


print(
    f"True Negatives : {tn}"
)


print(
    f"True Positives : {tp}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n==============================================")
print("             CONFUSION MATRIX")
print("==============================================")


print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n==============================================")
print("          CLASSIFICATION REPORT")
print("==============================================")


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
# SAVE RESULTS
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)


with open(
    OUTPUT_PATH,
    "w"
) as f:

    f.write(
        "FINAL MODEL EVALUATION\n"
    )

    f.write(
        "======================\n\n"
    )


    f.write(
        f"Model: {MODEL_PATH}\n"
    )


    f.write(
        f"Test samples: {len(y_test)}\n"
    )


    f.write(
        f"Selected Threshold: {threshold}\n\n"
    )


    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    f.write(
        f"ROC-AUC: {auc:.4f}\n\n"
    )


    # --------------------------------------------------------
    # Threshold Analysis
    # --------------------------------------------------------

    f.write(
        "THRESHOLD ANALYSIS\n"
    )

    f.write(
        "==================\n\n"
    )


    f.write(
        "Threshold | Accuracy | Precision | Recall | F1 | FP | TN\n"
    )

    f.write(
        "-" * 75 + "\n"
    )


    for result in threshold_results:

        f.write(
            f"{result['threshold']:8.2f} | "
            f"{result['accuracy']:8.4f} | "
            f"{result['precision']:9.4f} | "
            f"{result['recall']:6.4f} | "
            f"{result['f1']:6.4f} | "
            f"{result['fp']:2d} | "
            f"{result['tn']:3d}\n"
        )


    # --------------------------------------------------------
    # Threshold summary
    # --------------------------------------------------------

    f.write(
        "\n\nTHRESHOLD SUMMARY\n"
    )

    f.write(
        "=================\n\n"
    )


    f.write(
        f"Best Accuracy Threshold: "
        f"{best_accuracy['threshold']:.2f}\n"
    )

    f.write(
        f"Accuracy: "
        f"{best_accuracy['accuracy']:.4f}\n"
    )

    f.write(
        f"Precision: "
        f"{best_accuracy['precision']:.4f}\n"
    )

    f.write(
        f"Recall: "
        f"{best_accuracy['recall']:.4f}\n\n"
    )


    f.write(
        f"Best Precision Threshold: "
        f"{best_precision['threshold']:.2f}\n"
    )

    f.write(
        f"Accuracy: "
        f"{best_precision['accuracy']:.4f}\n"
    )

    f.write(
        f"Precision: "
        f"{best_precision['precision']:.4f}\n"
    )

    f.write(
        f"Recall: "
        f"{best_precision['recall']:.4f}\n\n"
    )


    f.write(
        f"Best F1 Threshold: "
        f"{best_f1['threshold']:.2f}\n"
    )

    f.write(
        f"Accuracy: "
        f"{best_f1['accuracy']:.4f}\n"
    )

    f.write(
        f"Precision: "
        f"{best_f1['precision']:.4f}\n"
    )

    f.write(
        f"Recall: "
        f"{best_f1['recall']:.4f}\n\n"
    )


    # --------------------------------------------------------
    # Final selected threshold results
    # --------------------------------------------------------

    f.write(
        "FINAL SELECTED THRESHOLD RESULTS\n"
    )

    f.write(
        "=================================\n\n"
    )


    f.write(
        f"Threshold: {threshold}\n\n"
    )


    f.write(
        f"Accuracy     : {accuracy:.4f}\n"
    )

    f.write(
        f"Precision    : {precision:.4f}\n"
    )

    f.write(
        f"Recall       : {recall:.4f}\n"
    )

    f.write(
        f"F1-Score     : {f1:.4f}\n"
    )

    f.write(
        f"ROC-AUC      : {auc:.4f}\n"
    )

    f.write(
        f"Specificity  : {specificity:.4f}\n"
    )

    f.write(
        f"False Positives: {fp}\n"
    )

    f.write(
        f"False Negatives: {fn}\n"
    )

    f.write(
        f"True Negatives : {tn}\n"
    )

    f.write(
        f"True Positives : {tp}\n\n"
    )


    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    f.write(
        "Confusion Matrix:\n"
    )

    f.write(
        str(cm)
    )

    f.write(
        "\n\n"
    )


    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    f.write(
        "Classification Report:\n"
    )

    f.write(
        report
    )


# ============================================================
# COMPLETED
# ============================================================

print("\n==============================================")
print("      EVALUATION COMPLETED SUCCESSFULLY")
print("==============================================")


print(
    "\nResults saved to:"
)


print(
    OUTPUT_PATH
)

