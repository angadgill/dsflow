"""
All helper functions for dsflow are in this file.

Author: Angad Gill
"""

"""
All plotting functions are in this file.

Author: Angad Gill
"""
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics


""" Metrics """


def gain(y_pred_proba, y_true):
    """
    Compute Gain defined as the sequence of True Positive Rate sorted by prediction probability in descending order.

    Returns: (gain, decile)
    """
    n = len(y_pred_proba)
    y_pred_proba = np.array(y_pred_proba)
    y_true = np.array(y_true)
    pred_proba_order_idx = np.argsort(y_pred_proba)[::-1]
    true_positives = np.cumsum(y_true[pred_proba_order_idx])
    g = true_positives/true_positives[-1]
    decile = np.arange(0, n)/(n-1)*10
    return g, decile


def gain_auc(y_pred_proba, y_test):
    """ Computes the area under the Gain curve """
    g, _ = gain(y_pred_proba, y_test)
    return g.sum()/len(g)


""" Plots """


def plot_roc_curve(y_pred_proba, y_true, legend=True, **kwargs):
    """ Plots a ROC curve """
    y_pred_proba = np.array(y_pred_proba)
    y_true = np.array(y_true)
    fpr, tpr, thres = metrics.roc_curve(y_true, y_pred_proba)
    plt.plot(fpr, tpr, **kwargs)
    plt.ylim(bottom=0, top=1.01)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    if legend:
        plt.legend()


def plot_precision_curve(y_pred_proba, y_true, legend=True, **kwargs):
    """ Plots a Precision curve """
    y_pred_proba = np.array(y_pred_proba)
    y_true = np.array(y_true)
    p, _, thres = metrics.precision_recall_curve(y_true, y_pred_proba)
    thres = np.append(thres, np.array(1))
    plt.plot(thres, p, **kwargs)
    plt.ylim(bottom=0, top=1.01)
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.title('Precision Curve')
    if legend:
        plt.legend()


def plot_recall_curve(y_pred_proba, y_true, legend=True, **kwargs):
    """ Plots a Recall curve """
    y_pred_proba = np.array(y_pred_proba)
    y_true = np.array(y_true)
    _, r, thres = metrics.precision_recall_curve(y_true, y_pred_proba)
    thres = np.append(thres, np.array(1))
    plt.plot(r, thres, **kwargs)
    plt.ylim(bottom=0, top=1.01)
    plt.xlabel('Threshold')
    plt.ylabel('Recall')
    plt.title('Recall Curve')
    if legend:
        plt.legend()


def plot_precision_recall_curve(y_pred_proba, y_true, legend=True, **kwargs):
    """ Plots a Precision-Recall curve """
    y_pred_proba = np.array(y_pred_proba)
    y_true = np.array(y_true)
    p, r, _ = metrics.precision_recall_curve(y_true, y_pred_proba)
    plt.plot(r, p, **kwargs)
    plt.ylim(bottom=0, top=1.01)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision - Recall Curve')
    if legend:
        plt.legend()


def plot_performance_curves(y_pred_proba, y_true, legend=True, **kwargs):
    """ Plots a 2x2 grid of Precision, Recall, Precision-Recall, and ROC curves """
    plt.subplot(2, 2, 1)
    plot_precision_curve(y_pred_proba, y_true, legend=False,  **kwargs)

    plt.subplot(2, 2, 2)
    plot_recall_curve(y_pred_proba, y_true, legend=False,  **kwargs)

    plt.subplot(2, 2, 3)
    plot_precision_recall_curve(y_pred_proba, y_true, legend=False,  **kwargs)

    plt.subplot(2, 2, 4)
    plot_roc_curve(y_pred_proba, y_true, legend=False,  **kwargs)

    plt.tight_layout()
    if legend:
        plt.legend()


def plot_gain_curve(y_pred_proba, y_true, legend=True, **kwargs):
    # n = len(y_pred_proba)
    g, decile = gain(y_pred_proba, y_true)
    # decile = np.arange(0, n)/(n-1)*10
    plt.plot(decile, g, **kwargs)
    plt.xlabel('Decile')
    plt.ylabel('Fraction of True Positive')
    plt.title('Gain Curve')
    if legend:
        plt.legend()