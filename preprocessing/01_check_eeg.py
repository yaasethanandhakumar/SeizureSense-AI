import os
import mne

RAW_PATH = r"C:\Projects\Epilepsy Seizure Prediction\clean_project\dataset\raw"

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

print("\n========== EEG DATASET CHECK ==========\n")

for filename in edf_files:

    file_path = os.path.join(RAW_PATH, filename)

    print("Reading:", filename)

    try:
        raw = mne.io.read_raw_edf(
            file_path,
            preload=False,
            verbose=False
        )

        print("Channels :", len(raw.ch_names))
        print("Sampling :", raw.info["sfreq"], "Hz")
        print("Duration :", round(raw.times[-1], 2), "seconds")
        print("Samples  :", raw.n_times)
        print("Status   : SUCCESS\n")

    except Exception as e:
        print("Status   : ERROR")
        print("Reason   :", e)
        print()

print("========== CHECK COMPLETED ==========")
