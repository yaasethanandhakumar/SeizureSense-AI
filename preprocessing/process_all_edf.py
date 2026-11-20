import os
import mne
import numpy as np
import pandas as pd

# Dataset folder
dataset_folder = r"C:\Projects\Epilepsy Seizure Prediction\dataset\chb01"

# Output folder for segments
output_folder = r"C:\Projects\Epilepsy Seizure Prediction\dataset\segments"
os.makedirs(output_folder, exist_ok=True)

segment_info = []
segment_number = 1

# 10-second window
window_seconds = 10

# Process every EDF file
for file in sorted(os.listdir(dataset_folder)):

    if file.endswith(".edf"):

        print(f"Processing {file}")

        file_path = os.path.join(dataset_folder, file)

        # Load EEG
        raw = mne.io.read_raw_edf(
            file_path,
            preload=True,
            verbose=False
        )

        # Band-pass filter
        raw.filter(
            0.5,
            40,
            verbose=False
        )

        # Get EEG data
        data = raw.get_data()

        # Sampling frequency
        sfreq = int(raw.info["sfreq"])

        window_size = window_seconds * sfreq

        # Create 10-second segments
        for start in range(
            0,
            data.shape[1] - window_size + 1,
            window_size
        ):

            end = start + window_size

            segment = data[:, start:end]

            filename = f"segment_{segment_number:05d}.npy"

            np.save(
                os.path.join(output_folder, filename),
                segment
            )

            segment_info.append([
                filename,
                file,
                start / sfreq,
                end / sfreq
            ])

            segment_number += 1

# Save metadata
df = pd.DataFrame(
    segment_info,
    columns=[
        "segment",
        "edf_file",
        "start_time",
        "end_time"
    ]
)

df.to_csv(
    r"C:\Projects\Epilepsy Seizure Prediction\dataset\segment_info.csv",
    index=False
)

print("\nCompleted Successfully")
print("Total Segments:", len(segment_info))