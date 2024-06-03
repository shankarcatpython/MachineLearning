import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Generate sample data
np.random.seed(42)
y_true = np.random.randint(2, size=100)  # Binary true labels
y_score = np.random.rand(100)  # Random scores for binary classification

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_score)

# Calculate Area Under the Curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')  # Random classifier line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

# Plot threshold values
for i, threshold in enumerate(thresholds):
    plt.text(fpr[i], tpr[i], '%0.2f' % threshold, fontsize=8, ha='center', va='bottom', color='red')

plt.show()
