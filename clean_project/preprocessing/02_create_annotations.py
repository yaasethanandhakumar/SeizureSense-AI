import json
import os

OUTPUT_PATH = r"C:\Projects\Epilepsy Seizure Prediction\clean_project\dataset\processed"

# Seizure information from chb01-summary.txt
seizures = {
    "chb01_03.edf": {
        "start": 2996,
        "end": 3036
    },
    "chb01_04.edf": {
        "start": 1467,
        "end": 1494
    },
    "chb01_15.edf": {
        "start": 1732,
        "end": 1772
    },
    "chb01_16.edf": {
        "start": 1015,
        "end": 1066
    },
    "chb01_18.edf": {
        "start": 1720,
        "end": 1810
    },
    "chb01_21.edf": {
        "start": 327,
        "end": 420
    },
    "chb01_26.edf": {
        "start": 1862,
        "end": 1963
    }
}

# All EDF files actually available
edf_files = [
    "chb01_01.edf",
    "chb01_02.edf",
    "chb01_03.edf",
    "chb01_04.edf",
    "chb01_15.edf",
    "chb01_16.edf",
    "chb01_18.edf",
    "chb01_21.edf",
    "chb01_26.edf"
]

annotations = {}

for filename in edf_files:

    if filename in seizures:

        seizure_start = seizures[filename]["start"]
        seizure_end = seizures[filename]["end"]

        # 30-minute preictal period
        preictal_start = max(0, seizure_start - 1800)

        annotations[filename] = {
            "preictal": {
                "start": preictal_start,
                "end": seizure_start
            },
            "seizure": {
                "start": seizure_start,
                "end": seizure_end
            }
        }

    else:

        annotations[filename] = {
            "preictal": None,
            "seizure": None
        }

os.makedirs(OUTPUT_PATH, exist_ok=True)

output_file = os.path.join(
    OUTPUT_PATH,
    "annotations.json"
)

with open(output_file, "w") as f:
    json.dump(annotations, f, indent=4)

print("\n========== ANNOTATIONS CREATED ==========\n")

for filename, data in annotations.items():

    print(filename)

    if data["preictal"] is not None:
        print(
            "Preictal:",
            data["preictal"]["start"],
            "to",
            data["preictal"]["end"],
            "seconds"
        )

        print(
            "Seizure :",
            data["seizure"]["start"],
            "to",
            data["seizure"]["end"],
            "seconds"
        )

    else:
        print("No seizure annotation")

    print()

print("Saved:")
print(output_file)