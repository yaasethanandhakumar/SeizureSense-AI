import mne
import matplotlib.pyplot as plt

file_path = r"C:\Projects\Epilepsy Seizure Prediction\dataset\chb01\chb01_01.edf"

raw = mne.io.read_raw_edf(file_path, preload=True)

print(raw)
print(raw.ch_names)
print(raw.info["sfreq"])

# Plot first 10 channels for 10 seconds
raw.plot(duration=10, n_channels=10, scalings="auto")

plt.show()