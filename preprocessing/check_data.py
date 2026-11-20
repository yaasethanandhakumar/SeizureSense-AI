import numpy as np

path = r"C:\Projects\Epilepsy Seizure Prediction\dataset"

X_train = np.load(path + r"\X_train_tf.npy")
y_train = np.load(path + r"\y_train.npy")

X_val = np.load(path + r"\X_val_tf.npy")
y_val = np.load(path + r"\y_val.npy")

print("TRAIN DATA")
print("Shape:", X_train.shape)
print("Minimum:", X_train.min())
print("Maximum:", X_train.max())
print("Mean:", X_train.mean())
print("Standard Deviation:", X_train.std())

print("\nVALIDATION DATA")
print("Shape:", X_val.shape)
print("Minimum:", X_val.min())
print("Maximum:", X_val.max())
print("Mean:", X_val.mean())
print("Standard Deviation:", X_val.std())

print("\nLABELS")
print("Training Normal:", np.sum(y_train == 0))
print("Training Preictal:", np.sum(y_train == 1))

print("Validation Normal:", np.sum(y_val == 0))
print("Validation Preictal:", np.sum(y_val == 1))