import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model import build_mlp_model
from tensorflow.keras.metrics import AUC

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train and evaluate MLP model for mood classification")
parser.add_argument('--run_name', type=str, required=True, help='Unique name for this run (used for logs/plots)')
parser.add_argument('--dropout1', type=float, default=0.3, help='Dropout rate for first hidden layer')
parser.add_argument('--dropout2', type=float, default=0.3, help='Dropout rate for second hidden layer')
parser.add_argument('--dropout3', type=float, default=0.2, help='Dropout rate for third hidden layer')
parser.add_argument('--epochs', type=int, default=50, help='Maximum number of training epochs')
parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
args = parser.parse_args()

# Set up directories
base_dir = "/mnt/data/Vineel/jamendo_project"
plots_dir = os.path.join(base_dir, "plots")
logs_dir = os.path.join(base_dir, "log")
model_output_path = os.path.join(base_dir, "models", f"model_{args.run_name}.keras")

os.makedirs(plots_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

# Logging setup
log_path = os.path.join(logs_dir, f"log_{args.run_name}.txt")
logging.basicConfig(filename=log_path, level=logging.INFO)

# Confirm GPU usage
gpus = tf.config.list_physical_devices('GPU')
print(f"Available devices: {gpus}")
logging.info(f"Available devices: {gpus}")

# Data paths
labels_path = os.path.join(base_dir, "labels", "moodtheme_labels.csv")
embeddings_dir = os.path.join(base_dir, "yamnet_embeddings_v3")

# Load CSV with .npy paths and multi-hot labels
df = pd.read_csv(labels_path)
X = np.array([np.load(os.path.join(embeddings_dir, p)) for p in df['path']])
y = df.iloc[:, 2:].values

# Data splits
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=42)

# Build model
model = build_mlp_model(
    input_dim=1024,
    output_dim=y.shape[1],
    dropout1=args.dropout1,
    dropout2=args.dropout2,
    dropout3=args.dropout3
)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=["accuracy", AUC(curve='PR', name='pr_auc')]
)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
]

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=args.epochs,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Evaluation
val_loss, val_acc, val_pr_auc = model.evaluate(X_val, y_val, verbose=1)
test_loss, test_acc, test_pr_auc = model.evaluate(X_test, y_test, verbose=1)

logging.info(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, PR_AUC: {val_pr_auc:.4f}")
logging.info(f"Test - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, PR_AUC: {test_pr_auc:.4f}")

# Save model
model.save(model_output_path)
print(f"Model saved to: {model_output_path}")
logging.info(f"Model saved to: {model_output_path}")

# Plotting
def save_plot(metric, title, ylabel):
    plt.figure()
    plt.plot(history.history[metric], label=f'Train {metric}')
    plt.plot(history.history[f'val_{metric}'], label=f'Val {metric}')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plot_file = os.path.join(plots_dir, f"{args.run_name}_{metric}.png")
    plt.savefig(plot_file)
    plt.close()

save_plot('loss', 'Training vs Validation Loss', 'Loss')
save_plot('accuracy', 'Training vs Validation Accuracy', 'Accuracy')
save_plot('pr_auc', 'Training vs Validation PR AUC', 'PR AUC')
