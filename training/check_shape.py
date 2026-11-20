import numpy as np

X_train = np.load(r"C:\Projects\Epilepsy Seizure Prediction\dataset\X_train.npy")

print("Original Shape:", X_train.shape)

X_train = np.transpose(X_train, (0, 2, 1))

print("TensorFlow Shape:", X_train.shape)