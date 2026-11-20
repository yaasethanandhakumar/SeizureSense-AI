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

DATA_PATH = os.path.join(BASE_PATH, "dataset")
MODEL_PATH = os.path.join(BASE_PATH, "saved_models")


# ============================================================
# CUSTOM ATTENTION LAYER
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

print("\n==============================================")
print("              LOADING TEST DATA")
print("==============================================")

X_test = np.load(
    os.path.join(DATA_PATH, "X_test.npy")
)

y_test = np.load(
    os.path.join(DATA_PATH, "y_test.npy")
)

# Convert:
# (samples, 23, 2560)
#       ↓
# (samples, 2560, 23)

if X_test.shape[1:] == (23, 2560):
    X_test = np.transpose(
        X_test,
        (0, 2, 1)
    )

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# MODELS TO COMPARE
# ============================================================

models = [
    "cnn_bilstm_attention.keras",
    "cnn_bilstm_attention_best.keras",
    "cnn_bilstm_attention_final.keras",
    "seizure_prediction_model.keras"
]


# ============================================================
# RESULTS STORAGE
# ============================================================

all_results = []


# ============================================================
# EVALUATE EACH MODEL
# ============================================================

for model_name in models:

    print("\n")
    print("==============================================")
    print("MODEL:", model_name)
    print("==============================================")

    model_file = os.path.join(
        MODEL_PATH,
        model_name
    )

    if not os.path.exists(model_file):

        print("Model file not found!")

        continue

    try:

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        model = tf.keras.models.load_model(
            model_file,
            custom_objects={
                "Attention": Attention
            },
            compile=False
        )

        print("Model loaded successfully.")

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        y_probability = model.predict(
            X_test,
            batch_size=16,
            verbose=0
        )

        y_probability = y_probability.ravel()

        # ----------------------------------------------------
        # Convert probability to class
        # ----------------------------------------------------

        y_pred = (
            y_probability >= 0.5
        ).astype(int)

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

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

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print("\nAccuracy :", f"{accuracy:.4f}")
        print("Precision:", f"{precision:.4f}")
        print("Recall   :", f"{recall:.4f}")
        print("F1-Score :", f"{f1:.4f}")
        print("ROC-AUC  :", f"{auc:.4f}")

        print("\nConfusion Matrix:")
        print(cm)

        print("\nPredictions:")
        print("Normal   :", np.sum(y_pred == 0))
        print("Preictal :", np.sum(y_pred == 1))

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        all_results.append([
            model_name,
            accuracy,
            precision,
            recall,
            f1,
            auc
        ])

        # Clear model from memory
        del model
        tf.keras.backend.clear_session()

    except Exception as e:

        print("\nERROR:")
        print(e)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n\n")
print("==============================================================")
print("                  MODEL COMPARISON")
print("==============================================================")

print(
    f"{'Model':40s}"
    f"{'Accuracy':>10s}"
    f"{'Precision':>11s}"
    f"{'Recall':>10s}"
    f"{'F1':>10s}"
    f"{'AUC':>10s}"
)

print("-" * 91)

for result in all_results:

    name, acc, prec, rec, f1, auc = result

    print(
        f"{name:40s}"
        f"{acc:10.4f}"
        f"{prec:11.4f}"
        f"{rec:10.4f}"
        f"{f1:10.4f}"
        f"{auc:10.4f}"
    )


# ============================================================
# IDENTIFY BEST MODEL
# ============================================================

if len(all_results) > 0:

    # Select based primarily on F1-score
    best_result = max(
        all_results,
        key=lambda x: x[4]
    )

    print("\n")
    print("==============================================================")
    print("                     BEST MODEL")
    print("==============================================================")

    print("Model    :", best_result[0])
    print("Accuracy :", f"{best_result[1]:.4f}")
    print("Precision:", f"{best_result[2]:.4f}")
    print("Recall   :", f"{best_result[3]:.4f}")
    print("F1-Score :", f"{best_result[4]:.4f}")
    print("ROC-AUC  :", f"{best_result[5]:.4f}")

else:

    print("\nNo models were successfully evaluated.")

print("\nEvaluation completed successfully!")