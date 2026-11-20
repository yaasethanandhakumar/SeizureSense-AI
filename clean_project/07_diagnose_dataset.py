import os
import numpy as np

# ============================================================
# DATASET DIAGNOSTIC
# ============================================================

BASE = r"C:\Projects\Epilepsy Seizure Prediction\clean_project"

DATA_DIR = os.path.join(
    BASE,
    "dataset",
    "final"
)

print("\n==========================================")
print("       EEG DATASET DIAGNOSTIC")
print("==========================================")

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# SHAPES
# ------------------------------------------------------------

print("\n========== SHAPES ==========")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val  :", X_val.shape)
print("y_val  :", y_val.shape)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)

# ------------------------------------------------------------
# LABEL DISTRIBUTION
# ------------------------------------------------------------

print("\n========== LABEL DISTRIBUTION ==========")

for name, y in [
    ("TRAIN", y_train),
    ("VALIDATION", y_val),
    ("TEST", y_test)
]:

    normal = np.sum(y == 0)
    preictal = np.sum(y == 1)

    print(f"\n{name}")
    print("Normal   :", normal)
    print("Preictal :", preictal)

# ------------------------------------------------------------
# DATA RANGE
# ------------------------------------------------------------

print("\n========== DATA RANGE ==========")

print("Training minimum :", np.min(X_train))
print("Training maximum :", np.max(X_train))
print("Training mean    :", np.mean(X_train))
print("Training std     :", np.std(X_train))

print("\nValidation minimum :", np.min(X_val))
print("Validation maximum :", np.max(X_val))
print("Validation mean    :", np.mean(X_val))
print("Validation std     :", np.std(X_val))

print("\nTest minimum :", np.min(X_test))
print("Test maximum :", np.max(X_test))
print("Test mean    :", np.mean(X_test))
print("Test std     :", np.std(X_test))

# ------------------------------------------------------------
# CHECK FOR NAN / INF
# ------------------------------------------------------------

print("\n========== NAN / INF CHECK ==========")

print(
    "Train NaN:",
    np.isnan(X_train).sum()
)

print(
    "Train Inf:",
    np.isinf(X_train).sum()
)

print(
    "Validation NaN:",
    np.isnan(X_val).sum()
)

print(
    "Validation Inf:",
    np.isinf(X_val).sum()
)

print(
    "Test NaN:",
    np.isnan(X_test).sum()
)

print(
    "Test Inf:",
    np.isinf(X_test).sum()
)

# ------------------------------------------------------------
# COMPARE NORMAL VS PREICTAL
# ------------------------------------------------------------

print("\n========== NORMAL VS PREICTAL ==========")

train_normal = X_train[y_train == 0]
train_preictal = X_train[y_train == 1]

print("\nNORMAL")

print(
    "Mean:",
    np.mean(train_normal)
)

print(
    "Std :",
    np.std(train_normal)
)

print(
    "Min :",
    np.min(train_normal)
)

print(
    "Max :",
    np.max(train_normal)
)

print("\nPREICTAL")

print(
    "Mean:",
    np.mean(train_preictal)
)

print(
    "Std :",
    np.std(train_preictal)
)

print(
    "Min :",
    np.min(train_preictal)
)

print(
    "Max :",
    np.max(train_preictal)
)

# ------------------------------------------------------------
# SAMPLE COMPARISON
# ------------------------------------------------------------

print("\n========== SAMPLE COMPARISON ==========")

normal_sample = train_normal[0]
preictal_sample = train_preictal[0]

print("\nNormal sample:")
print(
    "Mean:",
    np.mean(normal_sample)
)

print(
    "Std:",
    np.std(normal_sample)
)

print(
    "Min:",
    np.min(normal_sample)
)

print(
    "Max:",
    np.max(normal_sample)
)

print("\nPreictal sample:")
print(
    "Mean:",
    np.mean(preictal_sample)
)

print(
    "Std:",
    np.std(preictal_sample)
)

print(
    "Min:",
    np.min(preictal_sample)
)

print(
    "Max:",
    np.max(preictal_sample)
)

# ------------------------------------------------------------
# IDENTICAL SAMPLE CHECK
# ------------------------------------------------------------

print("\n========== DUPLICATE CHECK ==========")

print(
    "First two training samples identical:",
    np.array_equal(
        X_train[0],
        X_train[1]
    )
)

print(
    "First normal and first preictal identical:",
    np.array_equal(
        normal_sample,
        preictal_sample
    )
)

# ------------------------------------------------------------
# UNIQUE SAMPLE CHECK
# ------------------------------------------------------------

print("\n========== UNIQUE SAMPLE CHECK ==========")

print(
    "Training samples:",
    len(X_train)
)

print(
    "Unique training samples:",
    len(
        np.unique(
            X_train.reshape(
                len(X_train),
                -1
            ),
            axis=0
        )
    )
)

print("\n==========================================")
print("       DIAGNOSTIC COMPLETED")
print("==========================================")

