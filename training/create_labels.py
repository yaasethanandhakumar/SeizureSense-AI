import numpy as np

# Number of EEG segments
num_segments = 360

# Initially assume all are non-seizure
labels = np.zeros(num_segments)

# Example:
# Suppose seizure begins at segment 300

labels[270:300] = 1

print("Labels Created")

print("Normal Segments :", np.sum(labels == 0))
print("Preictal Segments :", np.sum(labels == 1))