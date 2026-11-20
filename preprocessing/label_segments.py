import pandas as pd
import os

# Paths
segment_info_path = r"C:\Projects\Epilepsy Seizure Prediction\dataset\segment_info.csv"
output_path = r"C:\Projects\Epilepsy Seizure Prediction\dataset\segment_labels.csv"

# Load segment information
df = pd.read_csv(segment_info_path)

# Seizure onset times from chb01-summary.txt
# Format: EDF file : seizure start time
seizures = {
    "chb01_03.edf": [2996],
    "chb01_04.edf": [1467],
    "chb01_15.edf": [1732],
    "chb01_16.edf": [1015],
    "chb01_18.edf": [1720],
    "chb01_21.edf": [327],
    "chb01_26.edf": [1862]
}

# Preictal duration = 30 minutes
PREICTAL_SECONDS = 30 * 60

# Default label: Normal
df["label"] = 0

# Label preictal segments
for i, row in df.iterrows():

    edf_file = row["edf_file"]
    start = row["start_time"]
    end = row["end_time"]

    if edf_file in seizures:

        for seizure_start in seizures[edf_file]:

            preictal_start = seizure_start - PREICTAL_SECONDS
            preictal_end = seizure_start

            # Segment overlaps the preictal period
            if start < preictal_end and end > preictal_start:
                df.at[i, "label"] = 1

# Save labels
df.to_csv(output_path, index=False)

# Display results
normal = (df["label"] == 0).sum()
preictal = (df["label"] == 1).sum()

print("\nLabeling Completed")
print("----------------------")
print("Total Segments :", len(df))
print("Normal Segments:", normal)
print("Preictal Segments:", preictal)
print("----------------------")

print("\nLabel file saved to:")
print(output_path)