from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

model = Sequential()

# First Convolution Layer
model.add(
    Conv1D(
        filters=32,
        kernel_size=3,
        activation="relu",
        input_shape=(2560, 23)
    )
)

# Pooling Layer
model.add(MaxPooling1D(pool_size=2))

# Convert to 1D vector
model.add(Flatten())

# Hidden Layer
model.add(Dense(64, activation="relu"))

# Output Layer
model.add(Dense(1, activation="sigmoid"))

model.summary()