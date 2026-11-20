import os
import mne
import numpy as np

# Load EEG file
file_path = r"C:\Projects\Epilepsy Seizure Prediction\dataset\chb01\chb01_01.edf"

raw = mne.io.read_raw_edf(file_path, preload=True)

# Apply Band-Pass Filter
raw.filter(0.5, 40)

# Get EEG data
data = raw.get_data()

sampling_rate = int(raw.info['sfreq'])

# 10-second window
window_size = 10 * sampling_rate

# Folder to save segments
output_folder = r"C:\Projects\Epilepsy Seizure Prediction\dataset\segments"
os.makedirs(output_folder, exist_ok=True)

segments = []

count = 1

for start in range(0, data.shape[1], window_size):

    if start + window_size <= data.shape[1]:

        segment = data[:, start:start + window_size]

        segments.append(segment)

        filename = os.path.join(output_folder, f"segment_{count:03d}.npy")
        np.save(filename, segment)

        count += 1

segments = np.array(segments)

print("Number of Segments:", len(segments))
print("Shape of Each Segment:", segments[0].shape)
print("Complete Dataset Shape:", segments.shape)
print("Saved to:", output_folder)