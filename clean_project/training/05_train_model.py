import os
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Projects\Epilepsy Seizure Prediction\clean_project"

DATA_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "final"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "saved_models"
)

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "cnn_bilstm_attention_final.keras"
)

# ============================================================
# LOAD DATA
# ============================================================

print("\n==========================================")
print("          LOADING DATA")
print("==========================================")

X_train = np.load(
    os.path.join(DATA_DIR, "X_train.npy")
)

X_val = np.load(
    os.path.join(DATA_DIR, "X_val.npy")
)

X_test = np.load(
    os.path.join(DATA_DIR, "X_test.npy")
)

y_train = np.load(
    os.path.join(DATA_DIR, "y_train.npy")
)

y_val = np.load(
    os.path.join(DATA_DIR, "y_val.npy")
)

y_test = np.load(
    os.path.join(DATA_DIR, "y_test.npy")
)

print("X_train :", X_train.shape)
print("X_val   :", X_val.shape)
print("X_test  :", X_test.shape)

print("y_train :", y_train.shape)
print("y_val   :", y_val.shape)
print("y_test  :", y_test.shape)

# ============================================================
# CHECK DATA
# ============================================================

print("\n==========================================")
print("          DATA DISTRIBUTION")
print("==========================================")

print(
    "Train Normal   :",
    np.sum(y_train == 0)
)

print(
    "Train Preictal :",
    np.sum(y_train == 1)
)

print(
    "Val Normal     :",
    np.sum(y_val == 0)
)

print(
    "Val Preictal   :",
    np.sum(y_val == 1)
)

print(
    "Test Normal    :",
    np.sum(y_test == 0)
)

print(
    "Test Preictal  :",
    np.sum(y_test == 1)
)

# ============================================================
# NORMALIZATION
# ============================================================
# Normalize each EEG segment independently.
# Shape:
# (samples, time, channels)

def normalize_eeg(X):

    X = X.astype(np.float32)

    mean = np.mean(
        X,
        axis=1,
        keepdims=True
    )

    std = np.std(
        X,
        axis=1,
        keepdims=True
    )

    std = np.maximum(
        std,
        1e-6
    )

    return (X - mean) / std


print("\n==========================================")
print("          NORMALIZING DATA")
print("==========================================")

X_train = normalize_eeg(X_train)
X_val = normalize_eeg(X_val)
X_test = normalize_eeg(X_test)

print(
    "Training mean :",
    np.mean(X_train)
)

print(
    "Training std  :",
    np.std(X_train)
)

print(
    "Validation mean :",
    np.mean(X_val)
)

print(
    "Validation std  :",
    np.std(X_val)
)

print(
    "Test mean :",
    np.mean(X_test)
)

print(
    "Test std  :",
    np.std(X_test)
)

# ============================================================
# CUSTOM ATTENTION LAYER
# ============================================================

@keras.saving.register_keras_serializable(
    package="Epilepsy"
)
class Attention(layers.Layer):

    def __init__(
        self,
        units=64,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.units = units

        self.W = layers.Dense(
            units,
            activation="tanh"
        )

        self.V = layers.Dense(
            1
        )

    def call(self, inputs):

        # inputs:
        # (batch, time, features)

        score = self.V(
            self.W(inputs)
        )

        weights = tf.nn.softmax(
            score,
            axis=1
        )

        context = tf.reduce_sum(
            inputs * weights,
            axis=1
        )

        return context

    def get_config(self):

        config = super().get_config()

        config.update({
            "units": self.units
        })

        return config

# ============================================================
# BUILD MODEL
# ============================================================

print("\n==========================================")
print("          BUILDING MODEL")
print("==========================================")

input_shape = (
    X_train.shape[1],
    X_train.shape[2]
)

inputs = keras.Input(
    shape=input_shape,
    name="EEG_Input"
)

# ------------------------------------------------------------
# CNN BLOCK 1
# ------------------------------------------------------------

x = layers.Conv1D(
    filters=32,
    kernel_size=3,
    padding="same",
    activation="relu",
    name="CNN_1"
)(inputs)

x = layers.BatchNormalization(
    name="BatchNorm_1"
)(x)

x = layers.MaxPooling1D(
    pool_size=2,
    name="MaxPool_1"
)(x)

# ------------------------------------------------------------
# CNN BLOCK 2
# ------------------------------------------------------------

x = layers.Conv1D(
    filters=64,
    kernel_size=3,
    padding="same",
    activation="relu",
    name="CNN_2"
)(x)

x = layers.BatchNormalization(
    name="BatchNorm_2"
)(x)

x = layers.MaxPooling1D(
    pool_size=2,
    name="MaxPool_2"
)(x)

# ------------------------------------------------------------
# BiLSTM
# ------------------------------------------------------------

x = layers.Bidirectional(
    layers.LSTM(
        64,
        return_sequences=True
    ),
    name="BiLSTM"
)(x)

# ------------------------------------------------------------
# ATTENTION
# ------------------------------------------------------------

x = Attention(
    units=64,
    name="Attention"
)(x)

# ------------------------------------------------------------
# DENSE
# ------------------------------------------------------------

x = layers.Dense(
    64,
    activation="relu",
    name="Dense_1"
)(x)

x = layers.Dropout(
    0.4,
    name="Dropout"
)(x)

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

outputs = layers.Dense(
    1,
    activation="sigmoid",
    name="Prediction"
)(x)

model = keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="CNN_BiLSTM_Attention"
)

# ============================================================
# CLASS WEIGHTS
# ============================================================

normal_count = np.sum(y_train == 0)
preictal_count = np.sum(y_train == 1)

total = normal_count + preictal_count

class_weight = {
    0: total / (2.0 * normal_count),
    1: total / (2.0 * preictal_count)
}

print("\nClass weights:")
print(class_weight)

# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=0.0005
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        keras.metrics.AUC(
            name="auc"
        ),
        keras.metrics.Precision(
            name="precision"
        ),
        keras.metrics.Recall(
            name="recall"
        )
    ]
)

# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n==========================================")
print("             MODEL SUMMARY")
print("==========================================")

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=7,
        restore_best_weights=True
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    ),

    keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_auc",
        mode="max",
        save_best_only=True
    )
]

# ============================================================
# TRAIN
# ============================================================

print("\n==========================================")
print("             STARTING TRAINING")
print("==========================================")

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=30,

    batch_size=16,

    class_weight=class_weight,

    callbacks=callbacks,

    verbose=1
)

# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n==========================================")
print("        LOADING BEST MODEL")
print("==========================================")

model = keras.models.load_model(
    MODEL_PATH
)

print("Best model loaded successfully.")

# ============================================================
# TEST PREDICTIONS
# ============================================================

print("\n==========================================")
print("        TESTING MODEL")
print("==========================================")

predictions = model.predict(
    X_test,
    batch_size=16,
    verbose=1
).reshape(-1)

# ============================================================
# PROBABILITY CHECK
# ============================================================

print("\n==========================================")
print("        PROBABILITY CHECK")
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

# ============================================================
# CLASS PREDICTIONS
# ============================================================

predicted_classes = (
    predictions >= 0.5
).astype(int)

print("\nPredicted Normal   :",
      np.sum(predicted_classes == 0))

print("Predicted Preictal :",
      np.sum(predicted_classes == 1))

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predicted_classes
)

print("\n==========================================")
print("        CONFUSION MATRIX")
print("==========================================")

print(cm)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n==========================================")
print("        CLASSIFICATION REPORT")
print("==========================================")

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
# AUC
# ============================================================

auc = roc_auc_score(
    y_test,
    predictions
)

print("\n==========================================")
print("             FINAL AUC")
print("==========================================")

print(
    "AUC :",
    round(auc, 4)
)

# ============================================================
# SAVE FINAL MODEL
# ============================================================

FINAL_MODEL = os.path.join(
    MODEL_DIR,
    "cnn_bilstm_attention_final.keras"
)

model.save(
    FINAL_MODEL
)

print("\n==========================================")
print("        TRAINING COMPLETED")
print("==========================================")

print(
    "Model saved at:"
)

print(FINAL_MODEL)