import mne
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load EEG
file_path = r"C:\Projects\Epilepsy Seizure Prediction\dataset\chb01\chb01_01.edf"

raw = mne.io.read_raw_edf(file_path, preload=True)

# Filter
raw.filter(0.5,40)

# Get data
data = raw.get_data()

# Normalize
scaler = StandardScaler()
data = scaler.fit_transform(data.T).T

# Parameters
sampling_rate = int(raw.info["sfreq"])
window_size = 10 * sampling_rate

segments = []

for start in range(0, data.shape[1]-window_size+1, window_size):

    segment = data[:, start:start+window_size]

    segments.append(segment)

segments = np.array(segments)

print("Before Transpose :", segments.shape)

X = np.transpose(segments,(0,2,1))

print("After Transpose :", X.shape)

labels = np.zeros(len(X))

labels[270:300] = 1

print(labels.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.2,
    random_state=42
)

print("Training Data :", X_train.shape)
print("Testing Data :", X_test.shape)