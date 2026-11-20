import numpy as np
import os

SEGMENT_DIR = r"C:\Projects\Epilepsy Seizure Prediction\clean_project\dataset\segments"
OUTPUT_DIR = r"C:\Projects\Epilepsy Seizure Prediction\clean_project\dataset\final"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n========== PREPARING FINAL DATASET ==========\n")

# Load segmented data
X = np.load(
    os.path.join(SEGMENT_DIR, "X_all.npy")
)

y = np.load(
    os.path.join(SEGMENT_DIR, "y_all.npy")
)

files = np.load(
    os.path.join(SEGMENT_DIR, "file_names.npy")
)

print("Original X shape:", X.shape)
print("Original y shape:", y.shape)

# ------------------------------------------------
# Convert:
# (samples, channels, time)
# to:
# (samples, time, channels)
# ------------------------------------------------

X = np.transpose(X, (0, 2, 1))

print("TensorFlow X shape:", X.shape)

# ------------------------------------------------
# Recording-level split
# ------------------------------------------------

train_files = [
    "chb01_01.edf",
    "chb01_02.edf",
    "chb01_03.edf",
    "chb01_04.edf",
    "chb01_15.edf"
]

val_files = [
    "chb01_16.edf",
    "chb01_18.edf"
]

test_files = [
    "chb01_21.edf",
    "chb01_26.edf"
]

train_mask = np.isin(files, train_files)
val_mask = np.isin(files, val_files)
test_mask = np.isin(files, test_files)

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

X_test = X[test_mask]
y_test = y[test_mask]

# ------------------------------------------------
# Save
# ------------------------------------------------

np.save(
    os.path.join(OUTPUT_DIR, "X_train.npy"),
    X_train
)

np.save(
    os.path.join(OUTPUT_DIR, "y_train.npy"),
    y_train
)

np.save(
    os.path.join(OUTPUT_DIR, "X_val.npy"),
    X_val
)

np.save(
    os.path.join(OUTPUT_DIR, "y_val.npy"),
    y_val
)

np.save(
    os.path.join(OUTPUT_DIR, "X_test.npy"),
    X_test
)

np.save(
    os.path.join(OUTPUT_DIR, "y_test.npy"),
    y_test
)

print("\n========== DATASET SPLIT ==========")

print("Training   :", X_train.shape)
print("Validation :", X_val.shape)
print("Testing    :", X_test.shape)

print("\nTraining labels:")
print(
    "Normal:",
    np.sum(y_train == 0),
    "Preictal:",
    np.sum(y_train == 1)
)

print("\nValidation labels:")
print(
    "Normal:",
    np.sum(y_val == 0),
    "Preictal:",
    np.sum(y_val == 1)
)

print("\nTesting labels:")
print(
    "Normal:",
    np.sum(y_test == 0),
    "Preictal:",
    np.sum(y_test == 1)
)

print("\n========== DATASET PREPARATION COMPLETED ==========")