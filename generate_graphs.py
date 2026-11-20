import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"C:\Projects\Epilepsy Seizure Prediction"

HISTORY_FILE = os.path.join(
    PROJECT_DIR,
    "saved_models",
    "cnn_bilstm_attention_corrected_history.npy"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "graphs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("        GENERATING MODEL PERFORMANCE GRAPHS")
print("=" * 60)

# ============================================================
# LOAD TRAINING HISTORY
# ============================================================

print("\nLoading training history...")

history = np.load(
    HISTORY_FILE,
    allow_pickle=True
).item()

print("History loaded successfully.")

print("\nAvailable metrics:")
for key in history.keys():
    print(" -", key)

# ============================================================
# HELPER FUNCTION
# ============================================================

def get_metric(history, names):
    for name in names:
        if name in history:
            return history[name]
    return None


# ============================================================
# GET METRICS
# ============================================================

accuracy = get_metric(
    history,
    ["accuracy", "binary_accuracy"]
)

val_accuracy = get_metric(
    history,
    ["val_accuracy", "val_binary_accuracy"]
)

loss = get_metric(
    history,
    ["loss"]
)

val_loss = get_metric(
    history,
    ["val_loss"]
)

auc = get_metric(
    history,
    ["auc"]
)

val_auc = get_metric(
    history,
    ["val_auc"]
)

precision = get_metric(
    history,
    ["precision"]
)

val_precision = get_metric(
    history,
    ["val_precision"]
)

recall = get_metric(
    history,
    ["recall"]
)

val_recall = get_metric(
    history,
    ["val_recall"]
)

# ============================================================
# NUMBER OF EPOCHS
# ============================================================

num_epochs = len(loss)

epochs = range(1, num_epochs + 1)

print("\nNumber of completed epochs:", num_epochs)

# ============================================================
# 1. ACCURACY GRAPH
# ============================================================

if accuracy is not None and val_accuracy is not None:

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        accuracy,
        marker="o",
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        val_accuracy,
        marker="o",
        label="Validation Accuracy"
    )

    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "accuracy_curve.png"
    )

    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved:", path)

# ============================================================
# 2. LOSS GRAPH
# ============================================================

if loss is not None and val_loss is not None:

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        loss,
        marker="o",
        label="Training Loss"
    )

    plt.plot(
        epochs,
        val_loss,
        marker="o",
        label="Validation Loss"
    )

    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "loss_curve.png"
    )

    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved:", path)

# ============================================================
# 3. AUC GRAPH
# ============================================================

if auc is not None and val_auc is not None:

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        auc,
        marker="o",
        label="Training AUC"
    )

    plt.plot(
        epochs,
        val_auc,
        marker="o",
        label="Validation AUC"
    )

    plt.title("Training and Validation ROC-AUC")
    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "auc_curve.png"
    )

    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved:", path)

# ============================================================
# 4. PRECISION GRAPH
# ============================================================

if precision is not None and val_precision is not None:

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        precision,
        marker="o",
        label="Training Precision"
    )

    plt.plot(
        epochs,
        val_precision,
        marker="o",
        label="Validation Precision"
    )

    plt.title("Training and Validation Precision")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "precision_curve.png"
    )

    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved:", path)

# ============================================================
# 5. RECALL GRAPH
# ============================================================

if recall is not None and val_recall is not None:

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        recall,
        marker="o",
        label="Training Recall"
    )

    plt.plot(
        epochs,
        val_recall,
        marker="o",
        label="Validation Recall"
    )

    plt.title("Training and Validation Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "recall_curve.png"
    )

    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved:", path)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("             GRAPH GENERATION COMPLETED")
print("=" * 60)

print("\nGraphs saved in:")

print(OUTPUT_DIR)

print("\nFiles:")

for file in os.listdir(OUTPUT_DIR):
    print(" -", file)

print("\nDone.")