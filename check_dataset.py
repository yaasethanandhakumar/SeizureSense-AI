import numpy as np
import os

DATA_PATH = r"C:\Projects\Epilepsy Seizure Prediction\dataset"


def check_data(name):

    X = np.load(
        os.path.join(DATA_PATH, name + ".npy")
    )

    print("\n========================================")
    print(name)
    print("========================================")

    print("Shape :", X.shape)
    print("Min   :", np.min(X))
    print("Max   :", np.max(X))
    print("Mean  :", np.mean(X))
    print("Std   :", np.std(X))

    print("NaN   :", np.isnan(X).sum())
    print("Inf   :", np.isinf(X).sum())


# ========================================
# Check X datasets
# ========================================

check_data("X_train")
check_data("X_val")
check_data("X_test")


# ========================================
# Check labels
# ========================================

print("\n========================================")
print("LABEL DISTRIBUTION")
print("========================================")

for name in ["y_train", "y_val", "y_test"]:

    y = np.load(
        os.path.join(DATA_PATH, name + ".npy")
    )

    normal = np.sum(y == 0)
    preictal = np.sum(y == 1)

    print("\n" + name)

    print("Total    :", len(y))
    print("Normal   :", normal)
    print("Preictal :", preictal)


# ========================================
# Class-wise statistics
# ========================================

print("\n========================================")
print("CLASS-WISE TRAINING STATISTICS")
print("========================================")

X_train = np.load(
    os.path.join(DATA_PATH, "X_train.npy")
)

y_train = np.load(
    os.path.join(DATA_PATH, "y_train.npy")
)

normal_data = X_train[y_train == 0]
preictal_data = X_train[y_train == 1]

print("\nNORMAL")

print("Mean :", np.mean(normal_data))
print("Std  :", np.std(normal_data))
print("Min  :", np.min(normal_data))
print("Max  :", np.max(normal_data))

print("\nPREICTAL")

print("Mean :", np.mean(preictal_data))
print("Std  :", np.std(preictal_data))
print("Min  :", np.min(preictal_data))
print("Max  :", np.max(preictal_data))

print("\n========================================")
print("DATASET CHECK COMPLETED")
print("========================================")