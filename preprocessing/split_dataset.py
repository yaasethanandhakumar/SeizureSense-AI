import numpy as np
from sklearn.model_selection import train_test_split

# Load dataset
X = np.load(r"C:\Projects\Epilepsy Seizure Prediction\dataset\X.npy")
y = np.load(r"C:\Projects\Epilepsy Seizure Prediction\dataset\y.npy")

print("Original Dataset")
print("X:", X.shape)
print("y:", y.shape)

# First split: 70% train, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Second split: temporary into 15% validation and 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDataset Split Completed")
print("----------------------")

print("Training:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nValidation:")
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("\nTesting:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nClass Distribution")

print("\nTraining")
print("Normal:", np.sum(y_train == 0))
print("Preictal:", np.sum(y_train == 1))

print("\nValidation")
print("Normal:", np.sum(y_val == 0))
print("Preictal:", np.sum(y_val == 1))

print("\nTesting")
print("Normal:", np.sum(y_test == 0))
print("Preictal:", np.sum(y_test == 1))

# Save datasets
dataset_path = r"C:\Projects\Epilepsy Seizure Prediction\dataset"

np.save(dataset_path + r"\X_train.npy", X_train)
np.save(dataset_path + r"\y_train.npy", y_train)

np.save(dataset_path + r"\X_val.npy", X_val)
np.save(dataset_path + r"\y_val.npy", y_val)

np.save(dataset_path + r"\X_test.npy", X_test)
np.save(dataset_path + r"\y_test.npy", y_test)

print("\nAll split files saved successfully.")