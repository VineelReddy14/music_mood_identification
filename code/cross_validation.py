import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from model import build_mlp_model
from tensorflow.keras.metrics import AUC

# Argument parsing
parser = argparse.ArgumentParser(description="Cross-validate MLP model for mood classification")
parser.add_argument('--run_name', type=str, required=True, help='Unique name for this run')
parser.add_argument('--folds', type=int, default=5, help='Number of cross-validation folds')
parser.add_argument('--epochs', type=int, default=50, help='Max epochs per fold')
parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
parser.add_argument('--dropout1', type=float, default=0.3)
parser.add_argument('--dropout2', type=float, default=0.3)
parser.add_argument('--dropout3', type=float, default=0.2)
args = parser.parse_args()

# Paths
base_dir = "/mnt/data/Vineel/jamendo_project"
plots_dir = os.path.join(base_dir, "plots")
logs_dir = os.path.join(base_dir, "log")
models_dir = os.path.join(base_dir, "models")
labels_path = os.path.join(base_dir, "labels", "moodtheme_labels.csv")
embeddings_dir = os.path.join(base_dir, "yamnet_embeddings_v3")

os.makedirs(plots_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Logging setup
log_file = os.path.join(logs_dir, f"log_{args.run_name}.txt")
validation_log_file = os.path.join(logs_dir, "validation_training_logs.txt")
results_log_file = os.path.join(logs_dir, "results.txt")
logging.basicConfig(filename=log_file, level=logging.INFO)
logging.info(f"Starting cross-validation run: {args.run_name}")

# Load data
df = pd.read_csv(labels_path)
X = np.array([np.load(os.path.join(embeddings_dir, p)) for p in df['path']])
y = df.iloc[:, 2:].values
y_strat = y.argmax(axis=1)  # For stratification

cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

# Store history for combined plotting
all_fold_metrics = {'loss': [], 'accuracy': [], 'pr_auc': []}
fold_results = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_strat), start=1):
    print(f"\n ##Training fold {fold}/{args.folds}")
    with open(validation_log_file, 'a') as f_log:
        f_log.write(f"Fold {fold}/{args.folds}\n")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

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
        metrics=['accuracy', AUC(curve='PR', name='pr_auc')]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    val_loss, val_acc, val_pr_auc = model.evaluate(X_val, y_val)
    fold_results.append((val_loss, val_acc, val_pr_auc))

    with open(results_log_file, 'a') as f_res:
        f_res.write(f"Fold {fold} - Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, PR_AUC: {val_pr_auc:.4f}\n")

    all_fold_metrics['loss'].append(history.history['val_loss'])
    all_fold_metrics['accuracy'].append(history.history['val_accuracy'])
    all_fold_metrics['pr_auc'].append(history.history['val_pr_auc'])

    model_path = os.path.join(models_dir, f"model_{args.run_name}_fold{fold}.keras")
    model.save(model_path)

# Plot all folds together for each metric
for metric in ['loss', 'accuracy', 'pr_auc']:
    plt.figure()
    for i, fold_vals in enumerate(all_fold_metrics[metric], start=1):
        plt.plot(fold_vals, label=f'Fold {i}')
    plt.title(f'{metric.upper()} Across Folds')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f'{args.run_name}_combined_{metric}.png'))
    plt.close()

# Write average results
mean_results = np.mean(fold_results, axis=0)
with open(results_log_file, 'a') as f:
    f.write(f"\nAverage Validation Metrics:\n")
    f.write(f"Loss: {mean_results[0]:.4f}, Accuracy: {mean_results[1]:.4f}, PR_AUC: {mean_results[2]:.4f}\n")

print("Cross-validation completed.")
