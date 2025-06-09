import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# Generate synthetic binary classification data
np.random.seed(42)
n_samples = 1000
# Random true binary labels with imbalance
y_true = np.random.binomial(1, 0.3, size=n_samples)
# Simulated predicted probabilities
y_scores = np.random.rand(n_samples) * 0.6 + y_true * 0.4  # higher scores for positives

# Compute precision, recall, and PR AUC
precision, recall, _ = precision_recall_curve(y_true, y_scores)
pr_auc = average_precision_score(y_true, y_scores)

# Plot Precision-Recall curve
plt.figure()
plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Example Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()