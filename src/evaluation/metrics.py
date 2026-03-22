"""
src/evaluation/metrics.py
Clinically meaningful metrics for skin cancer detection.

Primary metric: pAUC @ TPR >= 0.80
All metrics here are computed on HELD-OUT data only.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    classification_report, average_precision_score
)
from sklearn.calibration import calibration_curve
from pathlib import Path


def pauc_score(y_true: np.ndarray, y_prob: np.ndarray, min_tpr: float = 0.80) -> float:
    """
    Partial AUC normalized to [0, 1] at TPR >= min_tpr.

    Clinical motivation: We only care about the model's discriminative ability
    in the high-sensitivity operating region. A model that performs well at
    TPR < 0.80 but fails at TPR >= 0.80 is clinically dangerous.

    Args:
        y_true: Binary ground truth labels (0=benign, 1=malignant)
        y_prob: Predicted probability of malignancy
        min_tpr: Minimum true positive rate threshold (default 0.80)

    Returns:
        Normalized pAUC in [0, 1]
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # Only keep points where TPR >= min_tpr
    mask = tpr >= min_tpr
    if mask.sum() < 2:
        return 0.0

    restricted_fpr = fpr[mask]
    restricted_tpr = tpr[mask]

    # Normalize by the maximum possible area in the restricted region
    pauc_raw = auc(restricted_fpr, restricted_tpr)
    max_area = 1.0 - min_tpr
    return pauc_raw / max_area


def compute_all_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                         threshold: float = 0.5, min_tpr: float = 0.80) -> dict:
    """
    Compute the full suite of evaluation metrics.

    Args:
        y_true: Ground truth binary labels
        y_prob: Predicted probabilities
        threshold: Decision threshold for binary predictions
        min_tpr: pAUC lower TPR bound

    Returns:
        Dictionary of metric name → value
    """
    y_pred = (y_prob >= threshold).astype(int)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    full_auc = auc(fpr, tpr)
    pauc = pauc_score(y_true, y_prob, min_tpr=min_tpr)
    avg_precision = average_precision_score(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    ppv = tp / (tp + fp + 1e-9)
    npv = tn / (tn + fn + 1e-9)
    f1 = 2 * sensitivity * ppv / (sensitivity + ppv + 1e-9)

    return {
        f'pAUC (TPR>={min_tpr})': pauc,
        'AUC': full_auc,
        'Avg Precision (AP)': avg_precision,
        'Sensitivity (Recall)': sensitivity,
        'Specificity': specificity,
        'PPV (Precision)': ppv,
        'NPV': npv,
        'F1 Score': f1,
        'TP': int(tp),
        'FP': int(fp),
        'TN': int(tn),
        'FN': int(fn),
        'Threshold': threshold,
    }


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                             target_sensitivity: float = 0.85) -> float:
    """
    Find the lowest threshold that achieves the target sensitivity.

    Clinical motivation: For screening, we prioritize sensitivity.
    We want to catch >=85% of malignancies, then maximize specificity
    within that constraint.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # Find all thresholds achieving target sensitivity
    valid = thresholds[tpr >= target_sensitivity]
    if len(valid) == 0:
        return 0.5
    # Return the highest threshold (most selective) that still meets sensitivity
    return float(valid[-1])


def plot_evaluation_dashboard(y_true: np.ndarray, y_prob: np.ndarray,
                                model_name: str = 'Model',
                                save_dir: Path = None,
                                min_tpr: float = 0.80) -> None:
    """
    Full evaluation dashboard: ROC, PR curve, confusion matrix, reliability diagram.
    """
    threshold = find_optimal_threshold(y_true, y_prob, target_sensitivity=0.85)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = compute_all_metrics(y_true, y_prob, threshold=threshold, min_tpr=min_tpr)

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    BLUE = '#378ADD'
    CORAL = '#D85A30'
    TEAL = '#1D9E75'

    # ── Panel 1: ROC Curve with pAUC shaded ──
    ax1 = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax1.plot(fpr, tpr, color=BLUE, lw=2,
             label=f'ROC (AUC={metrics["AUC"]:.3f})')
    ax1.axhline(min_tpr, color=CORAL, linestyle='--', lw=1.5,
                label=f'TPR={min_tpr} threshold')
    mask = tpr >= min_tpr
    ax1.fill_between(fpr[mask], min_tpr, tpr[mask], alpha=0.2, color=CORAL,
                     label=f'pAUC region={metrics[f"pAUC (TPR>={min_tpr})"]:.3f}')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'ROC Curve — {model_name}')
    ax1.legend(fontsize=9)

    # ── Panel 2: Precision-Recall Curve ──
    ax2 = fig.add_subplot(gs[0, 1])
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ax2.plot(recall, precision, color=TEAL, lw=2,
             label=f'AP={metrics["Avg Precision (AP)"]:.3f}')
    baseline = y_true.mean()
    ax2.axhline(baseline, color='gray', linestyle='--', alpha=0.5,
                label=f'Baseline (prevalence={baseline:.3f})')
    ax2.set_xlabel('Recall (Sensitivity)')
    ax2.set_ylabel('Precision (PPV)')
    ax2.set_title(f'Precision-Recall Curve — {model_name}')
    ax2.legend(fontsize=9)

    # ── Panel 3: Confusion Matrix ──
    ax3 = fig.add_subplot(gs[0, 2])
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = ax3.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax3, fraction=0.046)
    labels = ['Benign', 'Malignant']
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, f'{cm[i,j]:,}\n({cm_norm[i,j]:.1%})',
                     ha='center', va='center',
                     color='white' if cm_norm[i, j] > 0.6 else 'black', fontsize=11)
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['Pred Benign', 'Pred Malignant'])
    ax3.set_yticklabels(['True Benign', 'True Malignant'])
    ax3.set_title(f'Confusion Matrix (t={threshold:.3f})')

    # ── Panel 4: Calibration (Reliability Diagram) ──
    ax4 = fig.add_subplot(gs[1, 0])
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    ax4.plot(prob_pred, prob_true, 's-', color=BLUE, lw=2, ms=6, label='Model')
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax4.fill_between([0, 1], [0, 1], [0, 1], alpha=0.05, color='gray')
    ax4.set_xlabel('Mean Predicted Probability')
    ax4.set_ylabel('Fraction of Positives')
    ax4.set_title(f'Reliability Diagram — {model_name}')
    ax4.legend(fontsize=9)

    # ── Panel 5: Score Distribution ──
    ax5 = fig.add_subplot(gs[1, 1])
    for label, name, color in [(0, 'Benign', BLUE), (1, 'Malignant', CORAL)]:
        vals = y_prob[y_true == label]
        ax5.hist(vals, bins=50, alpha=0.6, color=color, density=True, label=name)
    ax5.axvline(threshold, color='k', linestyle='--', lw=1.5,
                label=f'Decision threshold={threshold:.3f}')
    ax5.set_xlabel('Predicted Probability of Malignancy')
    ax5.set_ylabel('Density')
    ax5.set_title('Score Distribution by True Class')
    ax5.legend(fontsize=9)

    # ── Panel 6: Metrics Summary ──
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    key_metrics = {
        f'pAUC (TPR≥{min_tpr})': f'{metrics[f"pAUC (TPR>={min_tpr})"]:.4f}  ← PRIMARY',
        'AUC-ROC': f'{metrics["AUC"]:.4f}',
        'Avg Precision': f'{metrics["Avg Precision (AP)"]:.4f}',
        'Sensitivity': f'{metrics["Sensitivity (Recall)"]:.4f}',
        'Specificity': f'{metrics["Specificity"]:.4f}',
        'PPV': f'{metrics["PPV (Precision)"]:.4f}',
        'F1 Score': f'{metrics["F1 Score"]:.4f}',
        'True Positives': str(metrics['TP']),
        'False Negatives': str(metrics['FN']),
        'Threshold': f'{metrics["Threshold"]:.4f}',
    }
    table_data = [(k, v) for k, v in key_metrics.items()]
    table = ax6.table(cellText=table_data, colLabels=['Metric', 'Value'],
                      loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    # Highlight primary metric
    table[(1, 0)].set_facecolor('#E1F5EE')
    table[(1, 1)].set_facecolor('#E1F5EE')
    ax6.set_title(f'{model_name} — Metrics', fontsize=11)

    plt.suptitle(f'Evaluation Dashboard: {model_name}', fontsize=14, y=1.01)

    if save_dir:
        save_path = Path(save_dir) / f'eval_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')

    plt.show()
    return metrics


def print_metrics_table(metrics_dict: dict, model_name: str = 'Model'):
    """Pretty-print metrics to console."""
    print(f'\n{"="*50}')
    print(f'  {model_name}')
    print(f'{"="*50}')
    for k, v in metrics_dict.items():
        if isinstance(v, float):
            marker = ' ← PRIMARY' if 'pAUC' in k else ''
            print(f'  {k:<30} {v:.4f}{marker}')
        else:
            print(f'  {k:<30} {v}')
    print(f'{"="*50}\n')


if __name__ == '__main__':
    # Smoke test with synthetic data
    np.random.seed(42)
    n = 1000
    y_true = np.array([1]*35 + [0]*965)
    y_prob = np.where(y_true == 1,
                      np.random.beta(5, 2, n),
                      np.random.beta(1, 5, n))

    pauc = pauc_score(y_true, y_prob, min_tpr=0.80)
    print(f'pAUC @ TPR>=0.80: {pauc:.4f}')

    metrics = compute_all_metrics(y_true, y_prob)
    print_metrics_table(metrics, 'Test Model')
