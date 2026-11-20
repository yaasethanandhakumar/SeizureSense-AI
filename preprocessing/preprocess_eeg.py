import mne
import matplotlib.pyplot as plt

# Load EEG file
file_path = r"C:\Projects\Epilepsy Seizure Prediction\dataset\chb01\chb01_01.edf"

raw = mne.io.read_raw_edf(file_path, preload=True)

print("Original EEG Loaded")

# Apply Band-Pass Filter
raw_filtered = raw.copy().filter(l_freq=0.5, h_freq=40)

print("Filtering Completed")

# Display Filtered EEG
raw_filtered.plot(duration=10, n_channels=10, scalings="auto")

plt.show()544