import mne
import numpy as np

# Load EEG file
file_path = r"C:\Projects\Epilepsy Seizure Prediction\dataset\chb01\chb01_01.edf"

raw = mne.io.read_raw_edf(file_path, preload=True)

# Band-pass filter
raw.filter(0.5, 40)

# Get EEG data
data = raw.get_data()

# Z-score normalization
mean = np.mean(data, axis=1, keepdims=True)
std = np.std(data, axis=1, keepdims=True)

normalized_data = (data - mean) / std

print("Normalization Completed")
print("Original Shape:", data.shape)
print("Normalized Shape:", normalized_data.shape)
print("Mean:", np.mean(normalized_data))
print("Standard Deviation:", np.std(normalized_data))

np.save(
    r"C:\Projects\Epilepsy Seizure Prediction\dataset\normalized_chb01_01.npy",
    normalized_data
)

print("Normalized EEG saved successfully.")