import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Attention,
    GlobalAveragePooling1D,
    Dense,
    Dropout
)


# ============================================================
# CNN + BiLSTM + Attention Model
# Input shape:
# (time_steps, channels) = (2560, 23)
# ============================================================

input_layer = Input(
    shape=(2560, 23),
    name="EEG_Input"
)


# ============================================================
# CNN BLOCK 1
# ============================================================

x = Conv1D(
    filters=32,
    kernel_size=3,
    padding="same",
    activation="relu",
    name="CNN_1"
)(input_layer)

x = MaxPooling1D(
    pool_size=2,
    name="MaxPool_1"
)(x)


# ============================================================
# CNN BLOCK 2
# ============================================================

x = Conv1D(
    filters=64,
    kernel_size=3,
    padding="same",
    activation="relu",
    name="CNN_2"
)(x)

x = MaxPooling1D(
    pool_size=2,
    name="MaxPool_2"
)(x)


# ============================================================
# BiLSTM
# ============================================================

x = Bidirectional(
    LSTM(
        64,
        return_sequences=True
    ),
    name="BiLSTM"
)(x)


# ============================================================
# ATTENTION
# ============================================================

attention_output = Attention(
    name="Attention"
)([x, x])


# ============================================================
# GLOBAL AVERAGE POOLING
# ============================================================

x = GlobalAveragePooling1D(
    name="Global_Average_Pooling"
)(attention_output)


# ============================================================
# FULLY CONNECTED LAYER
# ============================================================

x = Dense(
    64,
    activation="relu",
    name="Dense_64"
)(x)


# ============================================================
# DROPOUT
# ============================================================

x = Dropout(
    0.5,
    name="Dropout"
)(x)


# ============================================================
# OUTPUT
#
# 0 = Normal
# 1 = Preictal
#
# Sigmoid returns probability of Preictal.
# ============================================================

output = Dense(
    1,
    activation="sigmoid",
    name="Seizure_Output"
)(x)


# ============================================================
# CREATE MODEL
# ============================================================

model = Model(
    inputs=input_layer,
    outputs=output,
    name="CNN_BiLSTM_Attention"
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()

