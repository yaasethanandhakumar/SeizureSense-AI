import numpy as np
from sklearn.model_selection import train_test_split

# Load dataset
X = np.load(r"C:\Projects\Epilepsy Seizure Prediction\dataset\X.npy")
y = np.load(r"C:\Projects\Epilepsy Seizure Prediction\dataset\y.npy")

print("Original Dataset")
print("X Shape:", X.shape)
print("y Shape:", y.shape)

# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Second split: 15% validation, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDataset Split Completed")
print("Training :", X_train.shape)
print("Validation :", X_val.shape)
print("Testing :", X_test.shape)

# Save datasets
np.save(r"C:\Projects\Epilepsy Seizure Prediction\dataset\X_train.npy", X_train)
np.save(r"C:\Projects\Epilepsy Seizure Prediction\dataset\X_val.npy", X_val)
np.save(r"C:\Projects\Epilepsy Seizure Prediction\dataset\X_test.npy", X_test)

np.save(r"C:\Projects\Epilepsy Seizure Prediction\dataset\y_train.npy", y_train)
np.save(r"C:\Projects\Epilepsy Seizure Prediction\dataset\y_val.npy", y_val)
np.save(r"C:\Projects\Epilepsy Seizure Prediction\dataset\y_test.npy", y_test)

print("\nAll datasets saved successfully.")