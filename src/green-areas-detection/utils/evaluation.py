import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def plot_history(training_history, metric='binary_accuracy', val=True, 
                 n_epochs=None, title='Binary accuracy'):
    """Plot the training history

    Parameters
    ----------
    training_history : dict
    metric : str
    val : bool
    n_epochs : int, optional
    title : str

    """
    if not n_epochs:
      n_epochs = len(training_history.history[metric])

    epochs = range(1,n_epochs+1)

    plt.plot(epochs, training_history.history[metric], label='train')
    if val:
        plt.plot(epochs, training_history.history[f'val_{metric}'], label='val')
    plt.legend()
    plt.grid()
    plt.xlabel('Number of epochs')
    plt.title(title)


def compute_pixelwise_retrieval_metrics(predicted_segmentations, ground_truth_masks):
    """Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Function taken from the patchore anomaly detection model
    https://github.com/amazon-science/patchcore-inspection/blob/main/src/patchcore/metrics.py

    Parameters
    ----------
    predicted_segmentations : np.ndarray [NxHxW]
        Contains generated segmentation masks.
    ground_truth_masks : np.ndarray [NxHxW]
        Contains predefined ground truth segmentation masks

    Returns
    -------
    dict
        {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }
    """

    if isinstance(predicted_segmentations, list):
        predicted_segmentations = np.stack(predicted_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_predicted_segmentations = predicted_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_predicted_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_predicted_segmentations
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_predicted_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_predicted_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }