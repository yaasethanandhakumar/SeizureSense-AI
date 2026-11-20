import os
import json
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
    Layer,
    LayerNormalization
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


# ============================================================
# 1. PATHS
# ============================================================

BASE_PATH = r"C:\Projects\Epilepsy Seizure Prediction"

DATA_PATH = os.path.join(
    BASE_PATH,
    "dataset"
)

SAVE_DIR = os.path.join(
    BASE_PATH,
    "saved_models"
)

os.makedirs(SAVE_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(
    SAVE_DIR,
    "cnn_bilstm_attention_corrected.keras"
)

FINAL_MODEL_PATH = os.path.join(
    SAVE_DIR,
    "cnn_bilstm_attention_corrected_final.keras"
)

HISTORY_PATH = os.path.join(
    SAVE_DIR,
    "cnn_bilstm_attention_corrected_history.npy"
)

NORMALIZATION_PATH = os.path.join(
    SAVE_DIR,
    "normalization_stats.json"
)


# ============================================================
# 2. RANDOM SEEDS
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\n==============================================")
print("             LOADING DATASET")
print("==============================================")

X_train = np.load(
    os.path.join(DATA_PATH, "X_train.npy")
)

X_val = np.load(
    os.path.join(DATA_PATH, "X_val.npy")
)

X_test = np.load(
    os.path.join(DATA_PATH, "X_test.npy")
)

y_train = np.load(
    os.path.join(DATA_PATH, "y_train.npy")
)

y_val = np.load(
    os.path.join(DATA_PATH, "y_val.npy")
)

y_test = np.load(
    os.path.join(DATA_PATH, "y_test.npy")
)


# ============================================================
# 4. DATA VALIDATION
# ============================================================

print("\n==============================================")
print("             DATA VALIDATION")
print("==============================================")

print("Original X_train:", X_train.shape)
print("Original X_val  :", X_val.shape)
print("Original X_test :", X_test.shape)

print("y_train:", y_train.shape)
print("y_val  :", y_val.shape)
print("y_test :", y_test.shape)

print(
    "y_train unique labels:",
    np.unique(y_train)
)

print(
    "y_val unique labels:",
    np.unique(y_val)
)

print(
    "y_test unique labels:",
    np.unique(y_test)
)


# ============================================================
# 5. INPUT SHAPE CORRECTION
# ============================================================

print("\n==============================================")
print("             INPUT SHAPE CHECK")
print("==============================================")


def correct_input_shape(X, name):

    if X.ndim != 3:

        raise ValueError(
            f"{name} must be 3-dimensional. "
            f"Received shape: {X.shape}"
        )

    # Expected original format:
    # (samples, channels, time)

    if X.shape[1:] == (23, 2560):

        print(
            f"{name}: transposing "
            "(samples, 23, 2560) -> "
            "(samples, 2560, 23)"
        )

        X = np.transpose(
            X,
            (0, 2, 1)
        )

    elif X.shape[1:] == (2560, 23):

        print(
            f"{name}: already in correct format"
        )

    else:

        raise ValueError(
            f"{name} has unexpected shape: {X.shape}"
        )

    return X.astype(
        np.float32
    )


X_train = correct_input_shape(
    X_train,
    "X_train"
)

X_val = correct_input_shape(
    X_val,
    "X_val"
)

X_test = correct_input_shape(
    X_test,
    "X_test"
)


print("\nFinal shapes:")

print(
    "X_train:",
    X_train.shape
)

print(
    "X_val  :",
    X_val.shape
)

print(
    "X_test :",
    X_test.shape
)


# ============================================================
# 6. LABEL DISTRIBUTION
# ============================================================

print("\n==============================================")
print("             LABEL DISTRIBUTION")
print("==============================================")


def print_distribution(name, y):

    normal = np.sum(y == 0)
    preictal = np.sum(y == 1)
    total = len(y)

    print(f"\n{name}")

    print(
        "Total    :",
        total
    )

    print(
        "Normal   :",
        normal
    )

    print(
        "Preictal :",
        preictal
    )

    print(
        "Normal % :",
        f"{normal / total:.4f}"
    )

    print(
        "Preictal %:",
        f"{preictal / total:.4f}"
    )


print_distribution(
    "TRAIN",
    y_train
)

print_distribution(
    "VALIDATION",
    y_val
)

print_distribution(
    "TEST",
    y_test
)


# ============================================================
# 7. NORMALIZATION
# ============================================================

print("\n==============================================")
print("          NORMALIZING DATA")
print("==============================================")


# IMPORTANT:
# Calculate statistics ONLY from training data.

training_mean = float(
    np.mean(X_train)
)

training_std = float(
    np.std(X_train)
)

print(
    "Training mean:",
    training_mean
)

print(
    "Training std :",
    training_std
)


# Avoid division by zero

if training_std == 0:

    raise ValueError(
        "Training standard deviation is zero."
    )


# Normalize all datasets using
# training statistics.

X_train = (
    X_train - training_mean
) / training_std

X_val = (
    X_val - training_mean
) / training_std

X_test = (
    X_test - training_mean
) / training_std


# Ensure float32

X_train = X_train.astype(
    np.float32
)

X_val = X_val.astype(
    np.float32
)

X_test = X_test.astype(
    np.float32
)


print("\nAfter normalization:")

print(
    "X_train mean:",
    np.mean(X_train)
)

print(
    "X_train std :",
    np.std(X_train)
)

print(
    "X_val mean  :",
    np.mean(X_val)
)

print(
    "X_val std   :",
    np.std(X_val)
)

print(
    "X_test mean :",
    np.mean(X_test)
)

print(
    "X_test std  :",
    np.std(X_test)
)


# ============================================================
# 8. SAVE NORMALIZATION STATISTICS
# ============================================================

normalization_stats = {

    "training_mean": training_mean,

    "training_std": training_std
}


with open(
    NORMALIZATION_PATH,
    "w"
) as f:

    json.dump(
        normalization_stats,
        f,
        indent=4
    )


print(
    "\nNormalization statistics saved to:"
)

print(
    NORMALIZATION_PATH
)


# ============================================================
# 9. CUSTOM ATTENTION LAYER
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="EEG"
)
class AttentionLayer(Layer):

    def __init__(
        self,
        **kwargs
    ):

        super().__init__(
            **kwargs
        )


    def build(
        self,
        input_shape
    ):

        self.W = self.add_weight(

            name="attention_weight",

            shape=(
                input_shape[-1],
                1
            ),

            initializer="glorot_uniform",

            trainable=True
        )


        self.b = self.add_weight(

            name="attention_bias",

            shape=(1,),

            initializer="zeros",

            trainable=True
        )


        super().build(
            input_shape
        )


    def call(
        self,
        inputs
    ):

        # Attention score

        score = tf.tanh(

            tf.matmul(
                inputs,
                self.W
            ) + self.b
        )


        # Attention weights

        weights = tf.nn.softmax(

            score,

            axis=1
        )


        # Weighted features

        context = (

            inputs * weights

        )


        # Weighted sum

        context = tf.reduce_sum(

            context,

            axis=1
        )


        return context


    def get_config(self):

        config = super().get_config()

        return config


# ============================================================
# 10. BUILD MODEL
# ============================================================

print("\n==============================================")
print("             BUILDING MODEL")
print("==============================================")


input_layer = Input(

    shape=(
        2560,
        23
    ),

    name="EEG_Input"
)


# ============================================================
# CNN BLOCK 1
# ============================================================

x = Conv1D(

    filters=32,

    kernel_size=7,

    padding="same",

    activation="relu"

)(input_layer)


x = BatchNormalization()(x)


x = MaxPooling1D(

    pool_size=2

)(x)


x = Dropout(

    0.20

)(x)


# ============================================================
# CNN BLOCK 2
# ============================================================

x = Conv1D(

    filters=64,

    kernel_size=5,

    padding="same",

    activation="relu"

)(x)


x = BatchNormalization()(x)


x = MaxPooling1D(

    pool_size=2

)(x)


x = Dropout(

    0.20

)(x)


# ============================================================
# CNN BLOCK 3
# ============================================================

x = Conv1D(

    filters=128,

    kernel_size=3,

    padding="same",

    activation="relu"

)(x)


x = BatchNormalization()(x)


x = MaxPooling1D(

    pool_size=2

)(x)


x = Dropout(

    0.20

)(x)


# ============================================================
# BiLSTM
# ============================================================

x = Bidirectional(

    LSTM(

        64,

        return_sequences=True,

        dropout=0.20

    )

)(x)


# ============================================================
# LAYER NORMALIZATION
# ============================================================

x = LayerNormalization()(x)


# ============================================================
# ATTENTION
# ============================================================

x = AttentionLayer(

    name="Attention"

)(x)


# ============================================================
# DENSE LAYER
# ============================================================

x = Dense(

    64,

    activation="relu"

)(x)


x = Dropout(

    0.40

)(x)


# ============================================================
# OUTPUT
# ============================================================

output_layer = Dense(

    1,

    activation="sigmoid",

    name="Prediction"

)(x)


# ============================================================
# CREATE MODEL
# ============================================================

model = Model(

    inputs=input_layer,

    outputs=output_layer

)


# ============================================================
# 11. COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=0.0001,

        clipnorm=1.0

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
# 12. MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# 13. CLASS WEIGHTS
# ============================================================

normal_count = np.sum(

    y_train == 0

)

preictal_count = np.sum(

    y_train == 1

)

total = (

    normal_count
    +
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


print("\n==============================================")
print("               CLASS WEIGHTS")
print("==============================================")

print(
    "Normal   :",
    class_weight[0]
)

print(
    "Preictal :",
    class_weight[1]
)


# ============================================================
# 14. CALLBACKS
# ============================================================

callbacks = [

    EarlyStopping(

        monitor="val_auc",

        mode="max",

        patience=7,

        restore_best_weights=True,

        verbose=1

    ),

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=3,

        min_lr=1e-6,

        verbose=1

    ),

    ModelCheckpoint(

        filepath=BEST_MODEL_PATH,

        monitor="val_auc",

        mode="max",

        save_best_only=True,

        verbose=1

    )

]


# ============================================================
# 15. TRAINING INFORMATION
# ============================================================

print("\n==============================================")
print("            STARTING TRAINING")
print("==============================================")

print(
    "Training samples:",
    len(X_train)
)

print(
    "Validation samples:",
    len(X_val)
)

print(
    "Test samples:",
    len(X_test)
)

print(
    "Batch size:",
    16
)

print(
    "Epochs:",
    30
)

print(
    "Learning rate:",
    0.0001
)


# ============================================================
# 16. TRAIN MODEL
# ============================================================

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
# 17. SAVE TRAINING HISTORY
# ============================================================

np.save(

    HISTORY_PATH,

    history.history,

    allow_pickle=True

)


print(
    "\nTraining history saved to:"
)

print(
    HISTORY_PATH
)


# ============================================================
# 18. LOAD BEST MODEL
# ============================================================

print("\n==============================================")
print("             LOADING BEST MODEL")
print("==============================================")


# IMPORTANT:
# custom_objects tells Keras how to recreate
# the custom AttentionLayer.

best_model = tf.keras.models.load_model(

    BEST_MODEL_PATH,

    custom_objects={

        "AttentionLayer":
        AttentionLayer

    },

    compile=True

)


print(
    "Best model loaded successfully."
)


# ============================================================
# 19. TEST BEST MODEL
# ============================================================

print("\n==============================================")
print("             TESTING BEST MODEL")
print("==============================================")


results = best_model.evaluate(

    X_test,

    y_test,

    verbose=1

)


print("\n==============================================")
print("              TEST RESULTS")
print("==============================================")


for name, value in zip(

    best_model.metrics_names,

    results

):

    print(

        f"{name}: {value:.4f}"

    )


# ============================================================
# 20. SAVE FINAL MODEL
# ============================================================

print("\n==============================================")
print("             SAVING FINAL MODEL")
print("==============================================")


best_model.save(

    FINAL_MODEL_PATH

)


print(
    "Final model saved to:"
)

print(
    FINAL_MODEL_PATH
)


# ============================================================
# 21. FINAL MESSAGE
# ============================================================

print("\n==============================================")
print("       TRAINING COMPLETED SUCCESSFULLY")
print("==============================================")


print(
    "\nBest model:"
)

print(
    BEST_MODEL_PATH
)


print(
    "\nFinal model:"
)

print(
    FINAL_MODEL_PATH
)


print(
    "\nNormalization statistics:"
)

print(
    NORMALIZATION_PATH
)


print(
    "\nTraining history:"
)

print(
    HISTORY_PATH
)


print("\n==============================================")
print("                  DONE")
print("==============================================")