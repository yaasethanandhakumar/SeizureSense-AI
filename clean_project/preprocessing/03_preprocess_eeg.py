import mne
import os
import glob

RAW_DIR = r"C:\Projects\Epilepsy Seizure Prediction\clean_project\dataset\raw"
OUTPUT_DIR = r"C:\Projects\Epilepsy Seizure Prediction\clean_project\dataset\filtered"

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = glob.glob(os.path.join(RAW_DIR, "*.edf"))

print("\n========== EEG PREPROCESSING ==========\n")

for file in files:

    filename = os.path.basename(file)
    name = os.path.splitext(filename)[0]

    print("Processing:", filename)

    raw = mne.io.read_raw_edf(
        file,
        preload=True,
        verbose=False
    )

    # Remove problematic recording-date metadata
    raw.set_meas_date(None)

    # Band-pass filter: 0.5–40 Hz
    raw.filter(
        l_freq=0.5,
        h_freq=40.0,
        verbose=False
    )

    # MNE raw files should end with raw.fif
    output_file = os.path.join(
        OUTPUT_DIR,
        name + "_filtered_raw.fif"
    )

    raw.save(
        output_file,
        overwrite=True,
        verbose=False
    )

    print("Saved:", output_file)
    print()

print("========== PREPROCESSING COMPLETED ==========")