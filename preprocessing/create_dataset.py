import os
import numpy as np
import pandas as pd

# Paths
segment_folder = r"C:\Projects\Epilepsy Seizure Prediction\dataset\segments"
labels_file = r"C:\Projects\Epilepsy Seizure Prediction\dataset\segment_labels.csv"

# Load labels
labels_df = pd.read_csv(labels_file)

X = []
y = []

missing = 0

print("Loading EEG segments...")

for _, row in labels_df.iterrows():

    segment_file = os.path.join(
        segment_folder,
        row["segment"]
    )

    if os.path.exists(segment_file):

        eeg = np.load(segment_file)

        X.append(eeg)
        y.append(int(row["label"]))

    else:
        print("Missing:", row["segment"])
        missing += 1

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

print("\nDataset Created Successfully")
print("----------------------------")
print("X Shape:", X.shape)
print("y Shape:", y.shape)
print("Missing Files:", missing)

print("\nClass Distribution")
print("Normal:", np.sum(y == 0))
print("Preictal:", np.sum(y == 1))

# Save
np.save(
    r"C:\Projects\Epilepsy Seizure Prediction\dataset\X.npy",
    X
)

np.save(
    r"C:\Projects\Epilepsy Seizure Prediction\dataset\y.npy",
    y
)

print("\nX.npy and y.npy saved successfully.")