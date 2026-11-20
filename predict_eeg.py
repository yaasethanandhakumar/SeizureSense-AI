import os
import json
import numpy as np
import mne
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from keras.saving import register_keras_serializable


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = r"C:\Projects\Epilepsy Seizure Prediction"

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "saved_models",
    "cnn_bilstm_attention_corrected.keras"
)

STATS_PATH = os.path.join(
    PROJECT_DIR,
    "saved_models",
    "normalization_stats.json"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs"
)

RESULT_FILE = os.path.join(
    OUTPUT_DIR,
    "eeg_prediction_results.txt"
)


# ============================================================
# EEG CONFIGURATION
# ============================================================

SAMPLING_RATE = 256
SEGMENT_DURATION = 10
SEGMENT_LENGTH = SAMPLING_RATE * SEGMENT_DURATION

LOW_FREQ = 0.5
HIGH_FREQ = 40.0


# ============================================================
# PREDICTION CONFIGURATION
# ============================================================

# Probability threshold for each 10-second segment
THRESHOLD = 0.88

# File-level threshold
#
# The complete EDF is classified as PREICTAL only when
# at least 5% of its segments are classified as PREICTAL.
#
# Example:
# 360 segments × 5% = 18 preictal segments required.
#
# This is a practical threshold for the current dashboard
# and should be validated later using the validation/test set.
MIN_PREICTAL_PERCENTAGE = 5.0


# ============================================================
# CUSTOM ATTENTION LAYER
# ============================================================

@register_keras_serializable(
    package="EEG",
    name="AttentionLayer"
)
class AttentionLayer(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):

        # ----------------------------------------------------
        # Saved model contains:
        #
        # attention weight -> (128, 1)
        # attention bias   -> (1,)
        # ----------------------------------------------------

        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )

        self.b = self.add_weight(
            name="attention_bias",
            shape=(1,),
            initializer="zeros",
            trainable=True
        )

        super().build(input_shape)

    def call(self, inputs):

        # Calculate attention score
        score = tf.tanh(
            tf.matmul(inputs, self.W) + self.b
        )

        # Calculate attention weights
        attention_weights = tf.nn.softmax(
            score,
            axis=1
        )

        # Apply attention
        context_vector = (
            inputs * attention_weights
        )

        # Sum over time dimension
        context_vector = tf.reduce_sum(
            context_vector,
            axis=1
        )

        return context_vector

    def get_config(self):

        config = super().get_config()

        return config


# ============================================================
# LOAD NORMALIZATION STATISTICS
# ============================================================

def load_normalization_stats():

    print("\n==========================================")
    print("       LOADING NORMALIZATION STATS")
    print("==========================================")

    if not os.path.exists(STATS_PATH):

        raise FileNotFoundError(
            "\nNormalization statistics file not found:\n"
            f"{STATS_PATH}"
        )

    with open(
        STATS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        stats = json.load(file)

    # --------------------------------------------------------
    # Find mean
    # --------------------------------------------------------

    if "training_mean" in stats:

        mean = stats["training_mean"]

    elif "mean" in stats:

        mean = stats["mean"]

    else:

        raise KeyError(
            "\nTraining mean was not found in "
            "normalization_stats.json"
        )

    # --------------------------------------------------------
    # Find standard deviation
    # --------------------------------------------------------

    if "training_std" in stats:

        std = stats["training_std"]

    elif "std" in stats:

        std = stats["std"]

    else:

        raise KeyError(
            "\nTraining standard deviation was not found "
            "in normalization_stats.json"
        )

    mean = float(mean)
    std = float(std)

    if std == 0:

        raise ValueError(
            "\nNormalization standard deviation is zero."
        )

    print(
        f"Training mean : {mean}"
    )

    print(
        f"Training std  : {std}"
    )

    return mean, std


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_prediction_model():

    print("\n==========================================")
    print("             LOADING MODEL")
    print("==========================================")

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "\nModel file not found:\n"
            f"{MODEL_PATH}"
        )

    print(
        f"Model path:\n{MODEL_PATH}"
    )

    print(
        "\nLoading custom AttentionLayer..."
    )

    # --------------------------------------------------------
    # The saved model uses:
    #
    # EEG>AttentionLayer
    #
    # Both possible names are supplied.
    # --------------------------------------------------------

    model = load_model(

        MODEL_PATH,

        custom_objects={

            "AttentionLayer":
                AttentionLayer,

            "EEG>AttentionLayer":
                AttentionLayer

        },

        compile=False
    )

    print(
        "\nModel loaded successfully."
    )

    print(
        "\nModel input shape :",
        model.input_shape
    )

    print(
        "Model output shape:",
        model.output_shape
    )

    print(
        "Total parameters  :",
        model.count_params()
    )

    return model


# ============================================================
# HANDLE DUPLICATE CHANNEL NAMES
# ============================================================

def rename_duplicate_channels(raw):

    original_channels = list(
        raw.ch_names
    )

    new_channels = []

    channel_counts = {}

    for channel in original_channels:

        clean_name = channel.strip()

        if clean_name not in channel_counts:

            channel_counts[
                clean_name
            ] = 0

            new_channels.append(
                clean_name
            )

        else:

            channel_counts[
                clean_name
            ] += 1

            duplicate_number = (
                channel_counts[
                    clean_name
                ]
            )

            new_name = (
                f"{clean_name}-"
                f"{duplicate_number - 1}"
            )

            new_channels.append(
                new_name
            )

    # --------------------------------------------------------
    # Create rename dictionary
    # --------------------------------------------------------

    rename_mapping = {}

    for old, new in zip(
        original_channels,
        new_channels
    ):

        if old != new:

            rename_mapping[old] = new

    # --------------------------------------------------------
    # Rename
    # --------------------------------------------------------

    if rename_mapping:

        raw.rename_channels(
            rename_mapping
        )

        print(
            "\nDuplicate channel names handled."
        )

        for old, new in rename_mapping.items():

            print(
                f"  {old} -> {new}"
            )

    return raw


# ============================================================
# PREPARE EEG CHANNELS
# ============================================================

def prepare_channels(raw):

    print("\n==========================================")
    print("          PREPARING EEG CHANNELS")
    print("==========================================")

    print(
        f"Original channel count: "
        f"{len(raw.ch_names)}"
    )

    # --------------------------------------------------------
    # Rename duplicate channels
    # --------------------------------------------------------

    raw = rename_duplicate_channels(
        raw
    )

    # --------------------------------------------------------
    # Expected 23-channel order
    #
    # IMPORTANT:
    # This must match the order used during training.
    # --------------------------------------------------------

    expected_channels = [

        "FP1-F7",
        "F7-T7",
        "T7-P7",
        "P7-O1",

        "FP1-F3",
        "F3-C3",
        "C3-P3",
        "P3-O1",

        "FP2-F4",
        "F4-C4",
        "C4-P4",
        "P4-O2",

        "FP2-F8",
        "F8-T8",
        "T8-P8-0",
        "P8-O2",

        "FZ-CZ",
        "CZ-PZ",

        "P7-T7",
        "T7-FT9",
        "FT9-FT10",
        "FT10-T8",
        "T8-P8-1"
    ]

    available_channels = list(
        raw.ch_names
    )

    # --------------------------------------------------------
    # Check missing channels
    # --------------------------------------------------------

    missing_channels = [

        channel

        for channel in expected_channels

        if channel not in available_channels

    ]

    if missing_channels:

        print(
            "\nWARNING: Missing expected channels:"
        )

        for channel in missing_channels:

            print(
                f"  - {channel}"
            )

        if len(
            available_channels
        ) < 23:

            raise ValueError(
                "\nThe EDF file contains fewer than "
                "23 channels."
            )

        print(
            "\nUsing first 23 available channels."
        )

        selected_channels = (
            available_channels[:23]
        )

    else:

        selected_channels = (
            expected_channels
        )

    # --------------------------------------------------------
    # Select channels
    # --------------------------------------------------------

    raw.pick(
        selected_channels
    )

    print(
        "\nSelected EEG channels:"
    )

    for index, channel in enumerate(
        raw.ch_names,
        start=1
    ):

        print(
            f"{index:2d}. {channel}"
        )

    # --------------------------------------------------------
    # Verify channel count
    # --------------------------------------------------------

    if len(raw.ch_names) != 23:

        raise ValueError(
            "\nChannel count mismatch.\n"
            f"Expected: 23\n"
            f"Found   : {len(raw.ch_names)}"
        )

    print(
        "\nFinal channel count: 23"
    )

    return raw


# ============================================================
# PREPROCESS EEG
# ============================================================

def preprocess_eeg(
    file_path,
    mean,
    std
):

    print("\n==========================================")
    print("          LOADING EEG FILE")
    print("==========================================")

    print(
        f"EDF file:\n{file_path}"
    )

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            "\nEDF file not found:\n"
            f"{file_path}"
        )

    # --------------------------------------------------------
    # Load EDF
    # --------------------------------------------------------

    raw = mne.io.read_raw_edf(
        file_path,
        preload=True,
        verbose=False
    )

    print(
        f"\nChannels      : "
        f"{len(raw.ch_names)}"
    )

    print(
        f"Sampling rate : "
        f"{raw.info['sfreq']} Hz"
    )

    print(
        f"Duration      : "
        f"{raw.times[-1]:.2f} seconds"
    )

    # --------------------------------------------------------
    # Prepare channels
    # --------------------------------------------------------

    raw = prepare_channels(
        raw
    )

    # --------------------------------------------------------
    # Check sampling rate
    # --------------------------------------------------------

    current_sampling_rate = (
        raw.info["sfreq"]
    )

    if current_sampling_rate != SAMPLING_RATE:

        print(
            f"\nWARNING: Sampling rate is "
            f"{current_sampling_rate} Hz."
        )

        print(
            f"Resampling to "
            f"{SAMPLING_RATE} Hz..."
        )

        raw.resample(
            SAMPLING_RATE,
            verbose=False
        )

    # --------------------------------------------------------
    # Bandpass filter
    # --------------------------------------------------------

    print(
        f"\nApplying bandpass filter: "
        f"{LOW_FREQ}–{HIGH_FREQ} Hz"
    )

    raw.filter(
        LOW_FREQ,
        HIGH_FREQ,
        verbose=False
    )

    # --------------------------------------------------------
    # Get EEG data
    # --------------------------------------------------------

    data = raw.get_data()

    print(
        "\nRaw EEG shape:",
        data.shape
    )

    n_channels, n_samples = (
        data.shape
    )

    # --------------------------------------------------------
    # Calculate complete segments
    # --------------------------------------------------------

    number_of_segments = (
        n_samples //
        SEGMENT_LENGTH
    )

    if number_of_segments == 0:

        raise ValueError(
            "\nEDF recording is shorter than "
            "one 10-second segment."
        )

    # --------------------------------------------------------
    # Keep complete segments only
    # --------------------------------------------------------

    usable_samples = (
        number_of_segments *
        SEGMENT_LENGTH
    )

    data = data[
        :,
        :usable_samples
    ]

    print(
        "\nNumber of segments:",
        number_of_segments
    )

    print(
        "Usable samples:",
        usable_samples
    )

    # --------------------------------------------------------
    # Reshape
    #
    # (23, samples)
    #       ↓
    # (23, segments, 2560)
    # --------------------------------------------------------

    segments = data.reshape(

        n_channels,
        number_of_segments,
        SEGMENT_LENGTH

    )

    # --------------------------------------------------------
    # Transpose
    #
    # Model expects:
    #
    # (segments, 2560, 23)
    # --------------------------------------------------------

    segments = np.transpose(

        segments,

        (1, 2, 0)

    )

    print(
        "\nSegment shape:",
        segments.shape
    )

    # --------------------------------------------------------
    # Verify shape
    # --------------------------------------------------------

    expected_shape = (

        number_of_segments,

        SEGMENT_LENGTH,

        23

    )

    if segments.shape != expected_shape:

        raise ValueError(

            "\nUnexpected EEG segment shape.\n"

            f"Expected: {expected_shape}\n"

            f"Got     : {segments.shape}"

        )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "          NORMALIZING EEG DATA"
    )

    print(
        "=========================================="
    )

    normalized = (

        segments - mean

    ) / std

    normalized = normalized.astype(
        np.float32
    )

    print(
        f"Normalized mean: "
        f"{np.mean(normalized):.6f}"
    )

    print(
        f"Normalized std : "
        f"{np.std(normalized):.6f}"
    )

    # --------------------------------------------------------
    # Final input verification
    # --------------------------------------------------------

    print(
        "\nModel input compatible."
    )

    print(
        "Final input shape:",
        normalized.shape
    )

    return normalized


# ============================================================
# PREDICT EEG SEGMENTS
# ============================================================

def predict_segments(
    model,
    segments
):

    print("\n==========================================")
    print("          RUNNING PREDICTIONS")
    print("==========================================")

    print(
        f"\nNumber of segments: "
        f"{len(segments)}"
    )

    print(
        f"Segment threshold: "
        f"{THRESHOLD}"
    )

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    probabilities = model.predict(

        segments,

        batch_size=16,

        verbose=1

    )

    # --------------------------------------------------------
    # Flatten output
    # --------------------------------------------------------

    probabilities = np.asarray(
        probabilities
    ).reshape(-1)

    # --------------------------------------------------------
    # Convert probability to class
    #
    # 0 = Normal
    # 1 = Preictal
    # --------------------------------------------------------

    predictions = (

        probabilities >= THRESHOLD

    ).astype(int)

    return (
        probabilities,
        predictions
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    probabilities,
    predictions
):

    total_segments = len(
        predictions
    )

    # --------------------------------------------------------
    # Count predictions
    # --------------------------------------------------------

    normal_count = int(
        np.sum(
            predictions == 0
        )
    )

    preictal_count = int(
        np.sum(
            predictions == 1
        )
    )

    # --------------------------------------------------------
    # Calculate percentages
    # --------------------------------------------------------

    normal_percentage = (

        normal_count /
        total_segments

    ) * 100

    preictal_percentage = (

        preictal_count /
        total_segments

    ) * 100

    # ========================================================
    # FILE-LEVEL DECISION
    # ========================================================

    if (

        preictal_percentage
        >= MIN_PREICTAL_PERCENTAGE

    ):

        overall_result = (
            "PREICTAL ACTIVITY DETECTED"
        )

    else:

        overall_result = (
            "NO PREICTAL ACTIVITY DETECTED"
        )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n")
    print("==========================================")
    print("             PREDICTION RESULTS")
    print("==========================================")

    print(
        f"\nTotal segments     : "
        f"{total_segments}"
    )

    print(
        f"Normal segments    : "
        f"{normal_count}"
    )

    print(
        f"Preictal segments  : "
        f"{preictal_count}"
    )

    print(
        f"\nNormal percentage  : "
        f"{normal_percentage:.2f}%"
    )

    print(
        f"Preictal percentage: "
        f"{preictal_percentage:.2f}%"
    )

    print(
        f"\nSegment threshold  : "
        f"{THRESHOLD:.2f}"
    )

    print(
        f"File-level threshold: "
        f"{MIN_PREICTAL_PERCENTAGE:.2f}%"
    )

    print(
        "\n------------------------------------------"
    )

    print(
        f"\nRESULT: {overall_result}"
    )

    print(
        "\n------------------------------------------"
    )

    # --------------------------------------------------------
    # Display preictal segments
    # --------------------------------------------------------

    preictal_indices = np.where(

        predictions == 1

    )[0]

    if len(
        preictal_indices
    ) > 0:

        print(
            "\nPreictal segment details:"
        )

        print(
            "------------------------------------------"
        )

        for index in preictal_indices:

            start_time = (

                index *
                SEGMENT_DURATION

            )

            end_time = (

                start_time +
                SEGMENT_DURATION

            )

            print(

                f"Segment {index + 1:03d} | "

                f"{start_time:6.0f}–"
                f"{end_time:6.0f} sec | "

                f"Probability = "
                f"{probabilities[index]:.4f} | "

                f"Prediction = PREICTAL"

            )

    else:

        print(
            "\nNo preictal segments detected."
        )

    return overall_result


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(

    file_path,

    probabilities,

    predictions,

    overall_result

):

    print(
        "\n=========================================="
    )

    print(
        "             SAVING RESULTS"
    )

    print(
        "=========================================="
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------------

    total_segments = len(
        predictions
    )

    normal_count = int(
        np.sum(
            predictions == 0
        )
    )

    preictal_count = int(
        np.sum(
            predictions == 1
        )
    )

    normal_percentage = (

        normal_count /
        total_segments

    ) * 100

    preictal_percentage = (

        preictal_count /
        total_segments

    ) * 100

    # ========================================================
    # WRITE RESULT FILE
    # ========================================================

    with open(

        RESULT_FILE,

        "w",

        encoding="utf-8"

    ) as file:

        file.write(
            "==========================================\n"
        )

        file.write(
            "       EEG SEIZURE PREDICTION RESULTS\n"
        )

        file.write(
            "==========================================\n\n"
        )

        # ----------------------------------------------------
        # File information
        # ----------------------------------------------------

        file.write(
            f"EDF File:\n{file_path}\n\n"
        )

        file.write(
            f"Model:\n{MODEL_PATH}\n\n"
        )

        # ----------------------------------------------------
        # EEG configuration
        # ----------------------------------------------------

        file.write(
            "Channels: 23\n"
        )

        file.write(
            f"Sampling Rate: "
            f"{SAMPLING_RATE} Hz\n"
        )

        file.write(
            f"Window Length: "
            f"{SEGMENT_DURATION} seconds\n"
        )

        file.write(
            f"Frequency Band: "
            f"{LOW_FREQ}-{HIGH_FREQ} Hz\n\n"
        )

        # ----------------------------------------------------
        # Thresholds
        # ----------------------------------------------------

        file.write(
            f"Segment Threshold: "
            f"{THRESHOLD}\n"
        )

        file.write(
            f"File-Level Threshold: "
            f"{MIN_PREICTAL_PERCENTAGE}%\n\n"
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        file.write(
            f"Total Segments: "
            f"{total_segments}\n"
        )

        file.write(
            f"Normal Segments: "
            f"{normal_count}\n"
        )

        file.write(
            f"Preictal Segments: "
            f"{preictal_count}\n"
        )

        file.write(
            f"Normal Percentage: "
            f"{normal_percentage:.2f}%\n"
        )

        file.write(
            f"Preictal Percentage: "
            f"{preictal_percentage:.2f}%\n\n"
        )

        # ----------------------------------------------------
        # IMPORTANT FOR STREAMLIT
        # ----------------------------------------------------

        file.write(
            f"RESULT: {overall_result}\n\n"
        )

        # ----------------------------------------------------
        # Segment-wise results
        # ----------------------------------------------------

        file.write(
            "==========================================\n"
        )

        file.write(
            "        SEGMENT-WISE PREDICTIONS\n"
        )

        file.write(
            "==========================================\n\n"
        )

        for index in range(
            total_segments
        ):

            start_time = (

                index *
                SEGMENT_DURATION

            )

            end_time = (

                start_time +
                SEGMENT_DURATION

            )

            if predictions[index] == 1:

                label = "PREICTAL"

            else:

                label = "NORMAL"

            file.write(

                f"Segment {index + 1:03d} | "

                f"{start_time:6.0f}–"
                f"{end_time:6.0f} sec | "

                f"Probability: "
                f"{probabilities[index]:.6f} | "

                f"Prediction: {label}\n"

            )

    print(
        "\nResults saved successfully:"
    )

    print(
        RESULT_FILE
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("\n")
    print("==========================================")
    print("       SEIZURESENSE AI PREDICTION")
    print("==========================================")

    print(
        "\nAttention-Based CNN-BiLSTM "
        "EEG Prediction System"
    )

    print(
        "=========================================="
    )

    try:

        # ====================================================
        # GET EDF FILE PATH
        # ====================================================

        file_path = input(

            "\nEnter the full path of the EDF file:\n"

        ).strip()

        # Remove quotation marks
        file_path = file_path.strip('"')
        file_path = file_path.strip("'")

        # ----------------------------------------------------
        # Check extension
        # ----------------------------------------------------

        if not file_path.lower().endswith(
            ".edf"
        ):

            raise ValueError(
                "\nPlease provide a valid EDF file."
            )

        # ====================================================
        # LOAD NORMALIZATION STATISTICS
        # ====================================================

        mean, std = (
            load_normalization_stats()
        )

        # ====================================================
        # LOAD MODEL
        # ====================================================

        model = (
            load_prediction_model()
        )

        # ====================================================
        # PREPROCESS EEG
        # ====================================================

        segments = (
            preprocess_eeg(
                file_path,
                mean,
                std
            )
        )

        # ====================================================
        # PREDICT
        # ====================================================

        probabilities, predictions = (

            predict_segments(

                model,

                segments

            )

        )

        # ====================================================
        # DISPLAY RESULTS
        # ====================================================

        overall_result = (

            display_results(

                probabilities,

                predictions

            )

        )

        # ====================================================
        # SAVE RESULTS
        # ====================================================

        save_results(

            file_path,

            probabilities,

            predictions,

            overall_result

        )

        # ====================================================
        # COMPLETE
        # ====================================================

        print(
            "\n=========================================="
        )

        print(
            "             PROCESS COMPLETE"
        )

        print(
            "=========================================="
        )

        print(
            "\nFinal prediction:"
        )

        print(
            f"RESULT: {overall_result}"
        )

        print(
            "\nOutput file:"
        )

        print(
            RESULT_FILE
        )

    except Exception as error:

        print(
            "\n=========================================="
        )

        print(
            "                 ERROR"
        )

        print(
            "=========================================="
        )

        print(
            f"\n{type(error).__name__}:"
        )

        print(
            error
        )

        print(
            "\nPrediction process stopped."
        )


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":

    main()