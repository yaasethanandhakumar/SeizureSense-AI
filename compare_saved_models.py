import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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

MODEL_DIR = os.path.join(
    BASE_PATH,
    "saved_models"
)

STATS_PATH = os.path.join(
    MODEL_DIR,
    "normalization_stats.json"
)


# ============================================================
# ATTENTION LAYER
# ============================================================

@keras.utils.register_keras_serializable(
    package="EEG"
)
class AttentionLayer(keras.layers.Layer):

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def build(
        self,
        input_shape
    ):

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

    def call(
        self,
        inputs
    ):

        score = tf.tanh(
            tf.matmul(
                inputs,
                self.attention_weight
            )
            +
            self.attention_bias
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

print(
    "Original X_test:",
    X_test.shape
)

print(
    "y_test:",
    y_test.shape
)


# ============================================================
# INPUT SHAPE
# ============================================================

if X_test.shape[1:] == (23, 2560):

    X_test = np.transpose(
        X_test,
        (0, 2, 1)
    )


print(
    "Final X_test:",
    X_test.shape
)


# ============================================================
# NORMALIZATION
# ============================================================

print("\n==============================================")
print("             NORMALIZATION")
print("==============================================")


with open(
    STATS_PATH,
    "r"
) as f:

    stats = json.load(f)


training_mean = float(
    stats["training_mean"]
)

training_std = float(
    stats["training_std"]
)


X_test = (
    X_test - training_mean
) / training_std


print(
    "Training mean:",
    training_mean
)

print(
    "Training std:",
    training_std
)

print(
    "Test mean:",
    np.mean(X_test)
)

print(
    "Test std:",
    np.std(X_test)
)


# ============================================================
# MODELS TO TEST
# ============================================================

model_names = [

    "cnn_bilstm_attention.keras",

    "cnn_bilstm_attention_best.keras",

    "cnn_bilstm_attention_final.keras",

    "cnn_bilstm_attention_corrected.keras",

    "cnn_bilstm_attention_corrected_best.keras",

    "cnn_bilstm_attention_corrected_final.keras",

    "seizure_prediction_model.keras"

]


# ============================================================
# RESULTS
# ============================================================

all_results = []


# ============================================================
# TEST EACH MODEL
# ============================================================

for model_name in model_names:

    model_path = os.path.join(
        MODEL_DIR,
        model_name
    )


    print("\n")
    print("=" * 60)
    print(
        "MODEL:",
        model_name
    )
    print("=" * 60)


    if not os.path.exists(model_path):

        print(
            "FILE NOT FOUND"
        )

        continue


    try:

        model = keras.models.load_model(
            model_path,
            custom_objects={
                "AttentionLayer": AttentionLayer
            },
            compile=False
        )

        print(
            "Model loaded successfully."
        )


    except Exception as e:

        print(
            "Could not load model."
        )

        print(
            str(e)
        )

        continue


    print(
        "Parameters:",
        model.count_params()
    )


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    probabilities = model.predict(
        X_test,
        batch_size=16,
        verbose=0
    ).ravel()


    # --------------------------------------------------------
    # PROBABILITY STATISTICS
    # --------------------------------------------------------

    print(
        "Probability min:",
        np.min(probabilities)
    )

    print(
        "Probability max:",
        np.max(probabilities)
    )

    print(
        "Probability mean:",
        np.mean(probabilities)
    )


    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    try:

        auc = roc_auc_score(
            y_test,
            probabilities
        )

    except Exception:

        auc = 0.0


    # --------------------------------------------------------
    # THRESHOLD 0.5
    # --------------------------------------------------------

    predictions = (
        probabilities >= 0.5
    ).astype(int)


    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )


    print("\nMetrics at threshold 0.50")

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1       : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {auc:.4f}"
    )


    print(
        "Confusion Matrix:"
    )

    print(cm)


    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    all_results.append(
        {
            "model": model_name,
            "auc": auc,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "min_probability": float(
                np.min(probabilities)
            ),
            "max_probability": float(
                np.max(probabilities)
            ),
            "mean_probability": float(
                np.mean(probabilities)
            )
        }
    )


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 80)
print("                    MODEL COMPARISON")
print("=" * 80)

print(
    f"{'Model':45s}"
    f"{'Accuracy':>10}"
    f"{'Precision':>11}"
    f"{'Recall':>10}"
    f"{'F1':>10}"
    f"{'AUC':>10}"
)

print("-" * 96)


for result in all_results:

    print(
        f"{result['model']:45s}"
        f"{result['accuracy']:10.4f}"
        f"{result['precision']:11.4f}"
        f"{result['recall']:10.4f}"
        f"{result['f1']:10.4f}"
        f"{result['auc']:10.4f}"
    )


# ============================================================
# BEST MODEL BY AUC
# ============================================================

if len(all_results) > 0:

    best = max(
        all_results,
        key=lambda x: x["auc"]
    )


    print("\n")
    print("=" * 60)
    print("             BEST MODEL BY ROC-AUC")
    print("=" * 60)

    print(
        "Model:",
        best["model"]
    )

    print(
        f"ROC-AUC: {best['auc']:.4f}"
    )

    print(
        f"Accuracy: {best['accuracy']:.4f}"
    )

    print(
        f"Precision: {best['precision']:.4f}"
    )

    print(
        f"Recall: {best['recall']:.4f}"
    )

    print(
        f"F1-Score: {best['f1']:.4f}"
    )


print("\n==============================================")
print("       MODEL COMPARISON COMPLETED")
print("==============================================")