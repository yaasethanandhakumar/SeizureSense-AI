import os
import json
import zipfile
import tempfile
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.layers import Layer


# ============================================================
# PATHS
# ============================================================

BASE_PATH = r"C:\Projects\Epilepsy Seizure Prediction"

MODEL_PATH = os.path.join(
    BASE_PATH,
    "saved_models",
    "cnn_bilstm_attention_corrected_best.keras"
)

DATA_PATH = os.path.join(
    BASE_PATH,
    "dataset"
)

NORMALIZATION_PATH = os.path.join(
    BASE_PATH,
    "saved_models",
    "normalization_stats.json"
)


# ============================================================
# CUSTOM ATTENTION LAYER
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="",
    name="AttentionLayer"
)
class AttentionLayer(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):

        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )

        self.b = self.add_weight(
            name="attention_bias",
            shape=(1,),
            initializer="zeros",
            trainable=True
        )

        super().build(input_shape)

    def call(self, inputs):

        score = tf.tanh(
            tf.matmul(inputs, self.W) + self.b
        )

        weights = tf.nn.softmax(
            score,
            axis=1
        )

        return tf.reduce_sum(
            inputs * weights,
            axis=1
        )

    def get_config(self):

        return super().get_config()


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\n==============================================")
print("             LOADING VALIDATION DATA")
print("==============================================")

X_val = np.load(
    os.path.join(DATA_PATH, "X_val.npy")
)

y_val = np.load(
    os.path.join(DATA_PATH, "y_val.npy")
)

print("Original X_val:", X_val.shape)
print("y_val:", y_val.shape)


# ============================================================
# INPUT SHAPE
# ============================================================

if X_val.shape[1:] == (23, 2560):

    X_val = np.transpose(
        X_val,
        (0, 2, 1)
    )

print("Final X_val:", X_val.shape)


# ============================================================
# NORMALIZATION
# ============================================================

print("\n==============================================")
print("             NORMALIZING DATA")
print("==============================================")

with open(
    NORMALIZATION_PATH,
    "r"
) as f:

    stats = json.load(f)

training_mean = stats["training_mean"]
training_std = stats["training_std"]

X_val = (
    X_val - training_mean
) / (
    training_std + 1e-8
)

print("Training mean:", training_mean)
print("Training std :", training_std)

print("Normalized validation mean:", np.mean(X_val))
print("Normalized validation std :", np.std(X_val))


# ============================================================
# BUILD EXACT ARCHITECTURE
# ============================================================

print("\n==============================================")
print("             BUILDING MODEL")
print("==============================================")

input_layer = tf.keras.Input(
    shape=(2560, 23),
    name="EEG_Input"
)

x = tf.keras.layers.Conv1D(
    32, 7, padding="same", activation="relu"
)(input_layer)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.MaxPooling1D(2)(x)

x = tf.keras.layers.Dropout(0.2)(x)


x = tf.keras.layers.Conv1D(
    64, 5, padding="same", activation="relu"
)(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.MaxPooling1D(2)(x)

x = tf.keras.layers.Dropout(0.2)(x)


x = tf.keras.layers.Conv1D(
    128, 3, padding="same", activation="relu"
)(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.MaxPooling1D(2)(x)

x = tf.keras.layers.Dropout(0.2)(x)


x = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(
        64,
        return_sequences=True,
        dropout=0.2
    )
)(x)

x = tf.keras.layers.LayerNormalization()(x)

x = AttentionLayer(
    name="Attention"
)(x)

x = tf.keras.layers.Dense(
    64,
    activation="relu"
)(x)

x = tf.keras.layers.Dropout(
    0.4
)(x)

output_layer = tf.keras.layers.Dense(
    1,
    activation="sigmoid",
    name="Prediction"
)(x)

model = tf.keras.Model(
    input_layer,
    output_layer
)

print("Model parameters:", model.count_params())


# ============================================================
# EXTRACT KERAS 3 WEIGHTS
# ============================================================

print("\n==============================================")
print("             EXTRACTING WEIGHTS")
print("==============================================")

temp_dir = tempfile.mkdtemp()

weights_path = os.path.join(
    temp_dir,
    "model.weights.h5"
)

with zipfile.ZipFile(
    MODEL_PATH,
    "r"
) as z:

    with z.open("model.weights.h5") as source:

        with open(
            weights_path,
            "wb"
        ) as target:

            target.write(
                source.read()
            )

print("Weights extracted.")


# ============================================================
# LOAD WEIGHTS USING KERAS 3 FORMAT
# ============================================================

print("\n==============================================")
print("             LOADING WEIGHTS")
print("==============================================")

model.load_weights(
    weights_path,
    skip_mismatch=False
)

print("Weights loaded successfully.")


# ============================================================
# PREDICTIONS
# ============================================================

print("\n==============================================")
print("             GENERATING PREDICTIONS")
print("==============================================")

y_prob = model.predict(
    X_val,
    batch_size=32,
    verbose=1
).ravel()


# ============================================================
# DEFAULT THRESHOLD
# ============================================================

THRESHOLD = 0.50

y_pred = (
    y_prob >= THRESHOLD
).astype(int)


# ============================================================
# METRICS
# ============================================================

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

auc = roc_auc_score(
    y_val,
    y_prob
)

cm = confusion_matrix(
    y_val,
    y_pred
)


# ============================================================
# RESULTS
# ============================================================

print("\n==============================================")
print("             VALIDATION RESULTS")
print("==============================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_val,
        y_pred,
        target_names=[
            "Normal",
            "Preictal"
        ],
        zero_division=0
    )
)


# ============================================================
# PROBABILITY STATISTICS
# ============================================================

print("\n==============================================")
print("          PROBABILITY STATISTICS")
print("==============================================")

print("Minimum:", np.min(y_prob))
print("Maximum:", np.max(y_prob))
print("Mean   :", np.mean(y_prob))
print("Median :", np.median(y_prob))


# ============================================================
# SAVE VALIDATION PROBABILITIES
# ============================================================

OUTPUT_PATH = os.path.join(
    BASE_PATH,
    "saved_models",
    "validation_predictions.npy"
)

np.save(
    OUTPUT_PATH,
    y_prob
)

print("\nValidation probabilities saved to:")
print(OUTPUT_PATH)

print("\n==============================================")
print("          VALIDATION COMPLETED")
print("==============================================")