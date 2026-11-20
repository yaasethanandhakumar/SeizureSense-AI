import mne
import numpy as np
import json
import os
import glob

FILTERED_DIR = r"C:\Projects\Epilepsy Seizure Prediction\clean_project\dataset\filtered"
ANNOTATION_FILE = r"C:\Projects\Epilepsy Seizure Prediction\clean_project\dataset\processed\annotations.json"
OUTPUT_DIR = r"C:\Projects\Epilepsy Seizure Prediction\clean_project\dataset\segments"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
SAMPLING_RATE = 256
SEGMENT_SECONDS = 10
SEGMENT_SAMPLES = SAMPLING_RATE * SEGMENT_SECONDS

# Load annotations
with open(ANNOTATION_FILE, "r") as f:
    annotations = json.load(f)

print("\n========== EEG SEGMENTATION ==========\n")

all_segments = []
all_labels = []
all_files = []

files = glob.glob(
    os.path.join(FILTERED_DIR, "*_filtered_raw.fif")
)

for file in files:

    filename = os.path.basename(file)

    # Convert filename to original EDF filename
    original_name = filename.replace(
        "_filtered_raw.fif",
        ".edf"
    )

    print("Processing:", original_name)

    raw = mne.io.read_raw_fif(
        file,
        preload=True,
        verbose=False
    )

    data = raw.get_data()

    total_samples = data.shape[1]

    total_segments = total_samples // SEGMENT_SAMPLES

    print("Channels :", data.shape[0])
    print("Samples  :", total_samples)
    print("Segments :", total_segments)

    # Get seizure annotation
    annotation = annotations.get(original_name)

    preictal_start = None
    preictal_end = None
    seizure_start = None
    seizure_end = None

    if annotation["preictal"] is not None:

        preictal_start = annotation["preictal"]["start"]
        preictal_end = annotation["preictal"]["end"]

        seizure_start = annotation["seizure"]["start"]
        seizure_end = annotation["seizure"]["end"]

    file_segments = []
    file_labels = []

    for i in range(total_segments):

        start_sample = i * SEGMENT_SAMPLES
        end_sample = start_sample + SEGMENT_SAMPLES

        segment = data[
            :,
            start_sample:end_sample
        ]

        start_time = i * SEGMENT_SECONDS
        end_time = start_time + SEGMENT_SECONDS

        # Default: Normal
        label = 0

        if preictal_start is not None:

            # Check whether segment is completely
            # inside the preictal period
            if (
                start_time >= preictal_start
                and end_time <= preictal_end
            ):
                label = 1

            # Exclude actual seizure period
            elif (
                start_time < seizure_end
                and end_time > seizure_start
            ):
                continue

        file_segments.append(segment)
        file_labels.append(label)

    file_segments = np.array(file_segments)
    file_labels = np.array(file_labels)

    print(
        "Normal segments  :",
        np.sum(file_labels == 0)
    )

    print(
        "Preictal segments:",
        np.sum(file_labels == 1)
    )

    # Save individual file
    segment_output = os.path.join(
        OUTPUT_DIR,
        original_name.replace(".edf", "_segments.npy")
    )

    label_output = os.path.join(
        OUTPUT_DIR,
        original_name.replace(".edf", "_labels.npy")
    )

    np.save(segment_output, file_segments)
    np.save(label_output, file_labels)

    all_segments.extend(file_segments)
    all_labels.extend(file_labels)
    all_files.extend(
        [original_name] * len(file_labels)
    )

    print()

# Convert to arrays
all_segments = np.array(all_segments)
all_labels = np.array(all_labels)

print("========== SEGMENTATION SUMMARY ==========")

print("Total segments :", len(all_segments))

print(
    "Normal         :",
    np.sum(all_labels == 0)
)

print(
    "Preictal       :",
    np.sum(all_labels == 1)
)

# Save combined dataset
np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_all.npy"
    ),
    all_segments
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_all.npy"
    ),
    all_labels
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "file_names.npy"
    ),
    np.array(all_files)
)

print("\nSaved:")
print(OUTPUT_DIR)

print("\n========== SEGMENTATION COMPLETED ==========")