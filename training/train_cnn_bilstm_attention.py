import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    BatchNormalization,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    Layer
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

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
# 1. PROJECT PATH
# ============================================================

BASE = r"C:\Projects\Epilepsy Seizure Prediction\clean_project"

DATA_DIR = os.path.join(
    BASE,
    "dataset",
    "final"
)

MODEL_DIR = os.path.join(
    BASE,
    "saved_models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "cnn_bilstm_attention_v3.keras"
)

# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n==========================================")
print("       LOADING EEG DATASET")
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

print("\nDataset shapes:")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val  :", X_val.shape)
print("y_val  :", y_val.shape)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)

# ============================================================
# 3. DATA VALIDATION
# ============================================================

if X_train.shape != (1786, 2560, 23):
    print("\nWARNING: Unexpected X_train shape:",
          X_train.shape)

if X_val.shape != (705, 2560, 23):
    print("\nWARNING: Unexpected X_val shape:",
          X_val.shape)

if X_test.shape != (571, 2560, 23):
    print("\nWARNING: Unexpected X_test shape:",
          X_test.shape)

print("\nLabel distribution:")

print(
    "Training Normal   :",
    np.sum(y_train == 0)
)

print(
    "Training Preictal :",
    np.sum(y_train == 1)
)

print(
    "Validation Normal   :",
    np.sum(y_val == 0)
)

print(
    "Validation Preictal :",
    np.sum(y_val == 1)
)

print(
    "Testing Normal   :",
    np.sum(y_test == 0)
)

print(
    "Testing Preictal :",
    np.sum(y_test == 1)
)

# ============================================================
# 4. NORMALIZATION
# ============================================================
# Normalize using training-set statistics only.

train_mean = np.mean(
    X_train,
    axis=(0, 1),
    keepdims=True
)

train_std = np.std(
    X_train,
    axis=(0, 1),
    keepdims=True
)

train_std = np.maximum(
    train_std,
    1e-8
)

X_train = (
    X_train - train_mean
) / train_std

X_val = (
    X_val - train_mean
) / train_std

X_test = (
    X_test - train_mean
) / train_std

print("\nNormalization completed.")

print(
    "Training mean:",
    np.mean(X_train)
)

print(
    "Training std :",
    np.std(X_train)
)

# ============================================================
# 5. CUSTOM ATTENTION LAYER
# ============================================================

class Attention(Layer):

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
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )

        super().build(input_shape)

    def call(self, x):

        score = tf.nn.tanh(
            tf.matmul(x, self.W) + self.b
        )

        attention_weights = tf.nn.softmax(
            score,
            axis=1
        )

        context_vector = (
            attention_weights * x
        )

        context_vector = tf.reduce_sum(
            context_vector,
            axis=1
        )

        return context_vector

    def get_config(self):

        config = super().get_config()

        return config

# ============================================================
# 6. BUILD MODEL
# ============================================================

print("\n==========================================")
print("          BUILDING MODEL")
print("==========================================")

input_layer = Input(
    shape=(2560, 23),
    name="EEG_Input"
)

# ------------------------------------------------------------
# CNN BLOCK 1
# ------------------------------------------------------------

x = Conv1D(
    filters=32,
    kernel_size=7,
    padding="same",
    activation="relu",
    name="CNN_1"
)(input_layer)

x = BatchNormalization(
    name="BatchNorm_1"
)(x)

x = MaxPooling1D(
    pool_size=2,
    name="MaxPool_1"
)(x)

# ------------------------------------------------------------
# CNN BLOCK 2
# ------------------------------------------------------------

x = Conv1D(
    filters=64,
    kernel_size=5,
    padding="same",
    activation="relu",
    name="CNN_2"
)(x)

x = BatchNormalization(
    name="BatchNorm_2"
)(x)

x = MaxPooling1D(
    pool_size=2,
    name="MaxPool_2"
)(x)

# ------------------------------------------------------------
# BiLSTM
# ------------------------------------------------------------

x = Bidirectional(
    LSTM(
        64,
        return_sequences=True,
        name="LSTM"
    ),
    name="BiLSTM"
)(x)

# ------------------------------------------------------------
# ATTENTION
# ------------------------------------------------------------

x = Attention(
    name="Attention"
)(x)

# ------------------------------------------------------------
# CLASSIFIER
# ------------------------------------------------------------

x = Dense(
    64,
    activation="relu",
    name="Dense_1"
)(x)

x = Dropout(
    0.4,
    name="Dropout"
)(x)

output_layer = Dense(
    1,
    activation="sigmoid",
    name="Prediction"
)(x)

# ============================================================
# 7. CREATE MODEL
# ============================================================

model = Model(
    inputs=input_layer,
    outputs=output_layer,
    name="CNN_BiLSTM_Attention"
)

# ============================================================
# 8. COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="binary_crossentropy",

    metrics=[

        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        ),

        tf.keras.metrics.AUC(
            name="auc"
        )
    ]
)

# ============================================================
# 9. MODEL SUMMARY
# ============================================================

print("\n==========================================")
print("             MODEL SUMMARY")
print("==========================================")

model.summary()

# ============================================================
# 10. CLASS WEIGHTS
# ============================================================

normal_count = np.sum(
    y_train == 0
)

preictal_count = np.sum(
    y_train == 1
)

total = (
    normal_count +
    preictal_count
)

class_weight = {

    0: total / (
        2 * normal_count
    ),

    1: total / (
        2 * preictal_count
    )
}

print("\n==========================================")
print("             CLASS WEIGHTS")
print("==========================================")

print(
    "Normal weight   :",
    class_weight[0]
)

print(
    "Preictal weight :",
    class_weight[1]
)

# ============================================================
# 11. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_auc",

    mode="max",

    patience=7,

    restore_best_weights=True,

    verbose=1
)

checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_auc",

    mode="max",

    save_best_only=True,

    verbose=1
)

reduce_lr = ReduceLROnPlateau(

    monitor="val_auc",

    mode="max",

    factor=0.5,

    patience=3,

    min_lr=1e-6,

    verbose=1
)

# ============================================================
# 12. TRAINING
# ============================================================

print("\n==========================================")
print("          TRAINING STARTED")
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

    callbacks=[
        early_stopping,
        checkpoint,
        reduce_lr
    ],

    verbose=1
)

# ============================================================
# 13. LOAD BEST MODEL
# ============================================================

print("\n==========================================")
print("          LOADING BEST MODEL")
print("==========================================")

best_model = tf.keras.models.load_model(

    MODEL_PATH,

    custom_objects={
        "Attention": Attention
    },

    safe_mode=False
)

# ============================================================
# 14. TEST
# ============================================================

print("\n==========================================")
print("             TESTING MODEL")
print("==========================================")

test_results = best_model.evaluate(

    X_test,

    y_test,

    verbose=1
)

print("\nRaw test results:")

for name, value in zip(
    best_model.metrics_names,
    test_results
):

    print(
        f"{name}: {value:.4f}"
    )

# ============================================================
# 15. GENERATE PREDICTIONS
# ============================================================

print("\n==========================================")
print("       GENERATING TEST PREDICTIONS")
print("==========================================")

probabilities = best_model.predict(
    X_test,
    verbose=1
).ravel()

predictions = (
    probabilities >= 0.5
).astype(int)

# ============================================================
# 16. METRICS
# ============================================================

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

auc = roc_auc_score(
    y_test,
    probabilities
)

print("\n==========================================")
print("             FINAL RESULTS")
print("==========================================")

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"AUC       : {auc:.4f}"
)

# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)

print("\n==========================================")
print("          CONFUSION MATRIX")
print("==========================================")

print(cm)

# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

print("\n==========================================")
print("       CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Normal",
            "Preictal"
        ],
        zero_division=0
    )
)

# ============================================================
# 19. PREDICTION DISTRIBUTION
# ============================================================

print("\n==========================================")
print("        PREDICTION DISTRIBUTION")
print("==========================================")

print(
    "Actual Normal    :",
    np.sum(y_test == 0)
)

print(
    "Actual Preictal  :",
    np.sum(y_test == 1)
)

print(
    "Predicted Normal :",
    np.sum(predictions == 0)
)

print(
    "Predicted Preictal:",
    np.sum(predictions == 1)
)

# ============================================================
# 20. SAMPLE PREDICTIONS
# ============================================================

print("\n==========================================")
print("           SAMPLE PREDICTIONS")
print("==========================================")

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
        if predictions[i] == 1
        else "Normal"
    )

    print(
        f"Sample {i+1:02d} | "
        f"Actual: {actual:<8} | "
        f"Predicted: {predicted:<8} | "
        f"Probability: {probabilities[i]:.4f}"
    )

# ============================================================
# 21. FINAL MESSAGE
# ============================================================

print("\n==========================================")
print(" CNN-BiLSTM-ATTENTION TRAINING COMPLETED")
print("==========================================")

print("\nBest model saved at:")

print(
    MODEL_PATH
)