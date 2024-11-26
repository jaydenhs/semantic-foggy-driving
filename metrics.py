import torch
from sklearn.metrics import jaccard_score

def calculate_mIoU(outputs, labels):
    """
    Calculates the mean Intersection over Union (mIoU) metric.
    Uses a weighted Jaccard core to account for label imbalance.
    """
    outputs = torch.argmax(outputs, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    mask = labels != 255
    outputs = outputs[mask]
    labels = labels[mask]
    mIoU = jaccard_score(labels.flatten(), outputs.flatten(), average="weighted")
    return mIoU