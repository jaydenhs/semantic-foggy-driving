import torch
import wandb
import numpy as np
from labels import map_trainId_to_label_name
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

def log_sample(input, input_name, label, output, split):
    input_img = input.permute(1, 2, 0).detach().cpu().numpy()
    label_img = label.detach().cpu().numpy()
    output_img = output.detach().cpu()
    
    output_img = torch.argmax(output_img, dim=0).numpy()

    wandb.log({
        f"{split}/input_image": wandb.Image(input_img, caption=f"Input Image: {input_name}"),
        f"{split}/label_image": wandb.Image(label_img, caption="Label Image", masks={
            "predictions": {
                "mask_data": label_img,
                "class_labels": {k: map_trainId_to_label_name(k) for k in np.unique(label_img)}
            }
        }),
        f"{split}/output_image": wandb.Image(output_img, caption="Output Image", masks={
            "predictions": {
                "mask_data": output_img,
                "class_labels": {k: map_trainId_to_label_name(k) for k in np.unique(output_img)}
            }
        })
    })
