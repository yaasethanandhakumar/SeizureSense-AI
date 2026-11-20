# SeizureSense AI

### Attention-Based Deep Learning Framework for Early Epileptic Seizure Prediction Using Multi-Channel EEG Signals

## Overview

SeizureSense AI is a deep learning-based application designed to identify preictal patterns in multi-channel Electroencephalogram (EEG) signals for early epileptic seizure prediction.

The framework combines Convolutional Neural Networks (CNN), Bidirectional Long Short-Term Memory (BiLSTM), and an Attention mechanism to learn spatial and temporal patterns from EEG signals.

## Key Features

- Multi-channel EEG signal processing
- EEG preprocessing and segmentation
- Preictal and interictal pattern classification
- CNN + BiLSTM + Attention architecture
- Model evaluation and threshold analysis
- Interactive Streamlit dashboard
- EEG prediction visualization

## Model Architecture

1. **CNN** – Extracts relevant features from EEG signals.
2. **BiLSTM** – Learns temporal dependencies in both forward and backward directions.
3. **Attention Mechanism** – Assigns importance to informative temporal features.
4. **Classification Layer** – Predicts the likelihood of a preictal state.

## Dataset

The project uses the CHB-MIT Scalp EEG Database.

The EEG recordings are processed into segments and categorized for model development.

**Note:** Original EEG recordings and processed datasets are not included in this repository due to their size and data distribution considerations.

## Technology Stack

- Python
- TensorFlow
- Keras
- MNE
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib
- Streamlit

## Project Structure

```text
SeizureSense-AI/
├── app.py
├── dashboard/
├── models/
├── preprocessing/
├── training/
├── evaluation/
├── clean_project/
├── outputs/
├── saved_models/
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/yaasethanandhakumar/SeizureSense-AI.git
cd SeizureSense-AI
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

**Note:** The trained model checkpoint and EEG datasets are not included. The required model must be obtained or generated before running inference.

## Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The previously evaluated model achieved a ROC-AUC of 94.10%. This result should be interpreted in the context of the dataset and evaluation procedure; it does not establish clinical performance.

Evaluation results and visualization plots are available in the `outputs` directory.

## Important Note

This project is developed for research and academic purposes. It is not a clinically validated seizure prediction or diagnostic system and should not be used for medical decision-making.

## Future Improvements

- Improve prediction precision and reduce false positives.
- Validate generalization across patients.
- Optimize real-time EEG processing.
- Expand evaluation using additional EEG recordings.

## Author

**Yaasetha N**

Biomedical Engineering | Computer Science Minor

**Project: SeizureSense AI**