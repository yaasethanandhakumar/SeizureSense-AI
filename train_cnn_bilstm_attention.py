import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    BatchNormalization,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    Layer
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ==============================
# 1. Load Dataset
# ==============================

DATA_PATH = r"C:\Projects\Epilepsy Seizure Prediction\dataset"

X_train = np.load(DATA_PATH + r"\X_train.npy")
X_val = np.load(DATA_PATH + r"\X_val.npy")
X_test = np.load(DATA_PATH + r"\X_test.npy")

X_train = np.transpose(X_train, (0, 2, 1))
X_val = np.transpose(X_val, (0, 2, 1))
X_test = np.transpose(X_test, (0, 2, 1))

y_train = np.load(DATA_PATH + r"\y_train.npy")
y_val = np.load(DATA_PATH + r"\y_val.npy")
y_test = np.load(DATA_PATH + r"\y_test.npy")

print("Dataset loaded successfully")

print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)

print("y_train:", y_train.shape)
print("y_val  :", y_val.shape)
print("y_test :", y_test.shape)


# ==============================
# 2. Attention Layer
# ==============================

class Attention(Layer):

    def __init__(self):
        super(Attention, self).__init__()

    def build(self, input_shape):

        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )

        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )

        super(Attention, self).build(input_shape)

    def call(self, x):

        score = tf.nn.tanh(tf.matmul(x, self.W) + self.b)

        attention_weights = tf.nn.softmax(
            score,
            axis=1
        )

        context_vector = attention_weights * x

        context_vector = tf.reduce_sum(
            context_vector,
            axis=1
        )

        return context_vector


# ==============================
# 3. Build Model
# ==============================

input_layer = Input(
    shape=(2560, 23),
    name="EEG_Input"
)


# CNN Block 1

x = Conv1D(
    filters=32,
    kernel_size=7,
    activation="relu",
    padding="same"
)(input_layer)

x = BatchNormalization()(x)

x = MaxPooling1D(
    pool_size=2
)(x)


# CNN Block 2

x = Conv1D(
    filters=64,
    kernel_size=5,
    activation="relu",
    padding="same"
)(x)

x = BatchNormalization()(x)

x = MaxPooling1D(
    pool_size=2
)(x)


# BiLSTM

x = Bidirectional(
    LSTM(
        64,
        return_sequences=True
    )
)(x)


# Attention

x = Attention()(x)


# Fully Connected Layer

x = Dense(
    64,
    activation="relu"
)(x)

x = Dropout(0.5)(x)


# Output

output_layer = Dense(
    1,
    activation="sigmoid",
    name="Prediction"
)(x)


# Create Model

model = Model(
    inputs=input_layer,
    outputs=output_layer
)


# ==============================
# 4. Compile Model
# ==============================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)


# ==============================
# 5. Display Model
# ==============================

model.summary()


# ==============================
# 6. Class Weights
# ==============================

normal_count = np.sum(y_train == 0)
preictal_count = np.sum(y_train == 1)

total = normal_count + preictal_count

class_weight = {
    0: total / (2 * normal_count),
    1: total / (2 * preictal_count)
}

print("\nClass Weights:")
print(class_weight)


# ==============================
# 7. Callbacks
# ==============================

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        r"C:\Projects\Epilepsy Seizure Prediction\saved_models\cnn_bilstm_attention.keras",
        monitor="val_auc",
        mode="max",
        save_best_only=True
    )
]


# ==============================
# 8. Train Model
# ==============================

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=30,

    batch_size=16,

    class_weight=class_weight,

    callbacks=callbacks,

    verbose=1
)


# ==============================
# 9. Test Model
# ==============================

print("\nEvaluating Test Dataset...")

results = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\nTest Results:")

for name, value in zip(
    model.metrics_names,
    results
):

    print(
        f"{name}: {value:.4f}"
    )


print("\nTraining completed successfully!")