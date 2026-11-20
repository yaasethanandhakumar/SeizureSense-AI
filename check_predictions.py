import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


# ============================================================
# PATHS
# ============================================================

BASE_PATH = r"C:\Projects\Epilepsy Seizure Prediction"

DATA_PATH = os.path.join(
    BASE_PATH,
    "dataset"
)

MODEL_PATH = os.path.join(
    BASE_PATH,
    "saved_models",
    "cnn_bilstm_attention.keras"
)


# ============================================================
# ATTENTION LAYER
# ============================================================

class Attention(Layer):

    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

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

        score = tf.nn.tanh(
            tf.matmul(x, self.W) + self.b
        )

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


# ============================================================
# LOAD TEST DATA
# ============================================================

X_test = np.load(
    os.path.join(DATA_PATH, "X_test.npy")
)

y_test = np.load(
    os.path.join(DATA_PATH, "y_test.npy")
)


# ============================================================
# CORRECT SHAPE
# ============================================================

if X_test.shape[1:] == (23, 2560):

    X_test = np.transpose(
        X_test,
        (0, 2, 1)
    )


print("\n========================================")
print("TEST DATA")
print("========================================")

print("Shape:", X_test.shape)
print("Labels:", y_test.shape)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n========================================")
print("LOADING MODEL")
print("========================================")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "Attention": Attention
    },
    compile=False
)

print("Model loaded successfully.")


# ============================================================
# GENERATE PROBABILITIES
# ============================================================

print("\n========================================")
print("GENERATING PREDICTIONS")
print("========================================")

probabilities = model.predict(
    X_test,
    batch_size=16,
    verbose=1
).ravel()


# ============================================================
# OVERALL PROBABILITY STATISTICS
# ============================================================

print("\n========================================")
print("OVERALL PROBABILITY STATISTICS")
print("========================================")

print("Minimum :", np.min(probabilities))
print("Maximum :", np.max(probabilities))
print("Mean    :", np.mean(probabilities))
print("Median  :", np.median(probabilities))
print("Std     :", np.std(probabilities))


# ============================================================
# CLASS-WISE PROBABILITY STATISTICS
# ============================================================

normal_prob = probabilities[y_test == 0]
preictal_prob = probabilities[y_test == 1]


print("\n========================================")
print("NORMAL PROBABILITIES")
print("========================================")

print("Count  :", len(normal_prob))
print("Mean   :", np.mean(normal_prob))
print("Median :", np.median(normal_prob))
print("Min    :", np.min(normal_prob))
print("Max    :", np.max(normal_prob))
print("Std    :", np.std(normal_prob))


print("\n========================================")
print("PREICTAL PROBABILITIES")
print("========================================")

print("Count  :", len(preictal_prob))
print("Mean   :", np.mean(preictal_prob))
print("Median :", np.median(preictal_prob))
print("Min    :", np.min(preictal_prob))
print("Max    :", np.max(preictal_prob))
print("Std    :", np.std(preictal_prob))


# ============================================================
# FIRST 20 PREDICTIONS
# ============================================================

print("\n========================================")
print("FIRST 20 PREDICTIONS")
print("========================================")

for i in range(min(20, len(probabilities))):

    actual = "Normal" if y_test[i] == 0 else "Preictal"

    print(
        f"{i+1:3d} | "
        f"Actual: {actual:8s} | "
        f"Probability: {probabilities[i]:.8f}"
    )


# ============================================================
# COMPLETION
# ============================================================

print("\n========================================")
print("DIAGNOSTIC CHECK COMPLETED")
print("========================================")