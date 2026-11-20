import os
import sys
import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_PATH = r"C:\Projects\Epilepsy Seizure Prediction"

DATASET_PATH = os.path.join(
    PROJECT_PATH,
    "dataset"
)

SAVED_MODEL_PATH = os.path.join(
    PROJECT_PATH,
    "saved_models",
    "seizure_prediction_model.keras"
)


# ============================================================
# ALLOW IMPORT FROM PROJECT ROOT
# ============================================================

if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)


# ============================================================
# IMPORT MODEL
# ============================================================

from models.cnn_bilstm_attention import model


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")


X_train = np.load(
    os.path.join(DATASET_PATH, "X_train.npy")
)

y_train = np.load(
    os.path.join(DATASET_PATH, "y_train.npy")
)

X_val = np.load(
    os.path.join(DATASET_PATH, "X_val.npy")
)

y_val = np.load(
    os.path.join(DATASET_PATH, "y_val.npy")
)

X_test = np.load(
    os.path.join(DATASET_PATH, "X_test.npy")
)

y_test = np.load(
    os.path.join(DATASET_PATH, "y_test.npy")
)


# ============================================================
# ORIGINAL SHAPES
# ============================================================

print("\nOriginal Shapes")

print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)

print("y_train:", y_train.shape)
print("y_val:", y_val.shape)
print("y_test:", y_test.shape)


# ============================================================
# CONVERT EEG DATA
#
# Original:
# (samples, channels, time)
#
# Required:
# (samples, time, channels)
# ============================================================

X_train = np.transpose(
    X_train,
    (0, 2, 1)
)

X_val = np.transpose(
    X_val,
    (0, 2, 1)
)

X_test = np.transpose(
    X_test,
    (0, 2, 1)
)


# ============================================================
# CONVERT LABELS TO FLOAT
#
# 0 = Normal
# 1 = Preictal
# ============================================================

y_train = y_train.astype(np.float32)
y_val = y_val.astype(np.float32)
y_test = y_test.astype(np.float32)


# ============================================================
# TENSORFLOW SHAPES
# ============================================================

print("\nTensorFlow Shapes")

print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)


# ============================================================
# CHECK DATA VALUES
# ============================================================

print("\nTRAINING DATA")

print("Minimum:", np.min(X_train))
print("Maximum:", np.max(X_train))
print("Mean:", np.mean(X_train))
print("Standard Deviation:", np.std(X_train))


print("\nVALIDATION DATA")

print("Minimum:", np.min(X_val))
print("Maximum:", np.max(X_val))
print("Mean:", np.mean(X_val))
print("Standard Deviation:", np.std(X_val))


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

train_normal = np.sum(y_train == 0)
train_preictal = np.sum(y_train == 1)

val_normal = np.sum(y_val == 0)
val_preictal = np.sum(y_val == 1)

test_normal = np.sum(y_test == 0)
test_preictal = np.sum(y_test == 1)


print("\nLABEL DISTRIBUTION")

print(
    "Training Normal:",
    train_normal
)

print(
    "Training Preictal:",
    train_preictal
)

print(
    "Validation Normal:",
    val_normal
)

print(
    "Validation Preictal:",
    val_preictal
)

print(
    "Testing Normal:",
    test_normal
)

print(
    "Testing Preictal:",
    test_preictal
)


# ============================================================
# CHECK LABEL VALUES
# ============================================================

unique_labels = np.unique(y_train)

print("\nUnique Training Labels:", unique_labels)

if not np.all(np.isin(unique_labels, [0, 1])):
    raise ValueError(
        "Labels must contain only 0 and 1."
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

classes = np.array([0, 1])

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train.astype(int)
)

class_weights = {
    0: float(weights[0]),
    1: float(weights[1])
}

print("\nClass Weights:")
print(class_weights)


# ============================================================
# CREATE SAVED MODEL DIRECTORY
# ============================================================

os.makedirs(
    os.path.dirname(SAVED_MODEL_PATH),
    exist_ok=True
)


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=SAVED_MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    mode="min",
    verbose=1
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

print("\nModel Architecture:")

model.summary()


# ============================================================
# TRAINING
# ============================================================

print("\nStarting Training...")


history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=20,

    batch_size=32,

    class_weight=class_weights,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# TRAINING COMPLETED
# ============================================================

print("\nTraining Completed!")

print(
    "\nBest model saved at:"
)

print(
    SAVED_MODEL_PATH
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\nEvaluating Validation Data...")


val_results = model.evaluate(
    X_val,
    y_val,
    verbose=1
)


print("\nValidation Results:")

for name, value in zip(
    model.metrics_names,
    val_results
):
    print(
        f"{name}: {value:.4f}"
    )


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\nEvaluating Test Data...")


test_results = model.evaluate(
    X_test,
    y_test,
    verbose=1
)


print("\nTest Results:")

for name, value in zip(
    model.metrics_names,
    test_results
):
    print(
        f"{name}: {value:.4f}"
    )


print("\n===================================")
print("TRAINING AND EVALUATION COMPLETED")
print("===================================")

