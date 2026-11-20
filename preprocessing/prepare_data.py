import numpy as np

dataset_path = r"C:\Projects\Epilepsy Seizure Prediction\dataset"

# Load datasets
X_train = np.load(dataset_path + r"\X_train.npy")
X_val = np.load(dataset_path + r"\X_val.npy")
X_test = np.load(dataset_path + r"\X_test.npy")

y_train = np.load(dataset_path + r"\y_train.npy")
y_val = np.load(dataset_path + r"\y_val.npy")
y_test = np.load(dataset_path + r"\y_test.npy")

print("Original Shapes")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)

# Change shape:
# (samples, channels, time)
#          ↓
# (samples, time, channels)

X_train = np.transpose(X_train, (0, 2, 1))
X_val = np.transpose(X_val, (0, 2, 1))
X_test = np.transpose(X_test, (0, 2, 1))

print("\nTensorFlow Shapes")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)

# Save prepared data
np.save(dataset_path + r"\X_train_tf.npy", X_train)
np.save(dataset_path + r"\X_val_tf.npy", X_val)
np.save(dataset_path + r"\X_test_tf.npy", X_test)

print("\nData preparation completed successfully.")