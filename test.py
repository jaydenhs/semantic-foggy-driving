import os
import numpy as np
import torch
import wandb
import random
from tqdm.auto import tqdm
from sklearn.metrics import jaccard_score
from models.unet import UNet
from dataset import get_data, make_dataloader

# Set random seeds
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)
torch.cuda.empty_cache()

wandb.login()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model_and_config(artifact_name, model_class, device):
    """
    Loads the model checkpoint and configuration from a W&B artifact.
    """
    artifact = wandb.use_artifact(artifact_name, type="model")
    artifact_dir = artifact.download()

    # Load configuration metadata
    config = artifact.metadata
    if not config:
        raise ValueError("Artifact metadata (configuration) is missing.")
    
    # Find the checkpoint file automatically
    checkpoint_files = [f for f in os.listdir(artifact_dir) if f.endswith(".pth")]
    if not checkpoint_files:
        raise FileNotFoundError("No .pth file found in the artifact directory.")
    model_path = os.path.join(artifact_dir, checkpoint_files[0])

    # Initialize and load the model
    model = model_class(in_channels=3, num_classes=19, base_channels=config["base_channels"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    return model, config


def evaluate_model(model, test_loader):
    """
    Evaluates the model on the test dataset and computes the mIoU.
    """
    model.eval()
    total_mIoU = 0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for sample in pbar:
            inputs = sample["input_image"].permute(0, 3, 1, 2).to(device)
            labels = sample["target_image"].to(device)
            outputs = model(inputs)
            mIoU = calculate_mIoU(outputs, labels)
            total_mIoU += mIoU

            pbar.set_postfix({"mIoU": mIoU})
    avg_mIoU = total_mIoU / len(test_loader)
    return avg_mIoU

def calculate_mIoU(outputs, labels):
    """
    Calculates the mean Intersection over Union (mIoU) metric.
    """
    outputs = torch.argmax(outputs, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    mask = labels != 255
    outputs = outputs[mask]
    labels = labels[mask]
    mIoU = jaccard_score(labels.flatten(), outputs.flatten(), average="weighted")
    return mIoU


if __name__ == "__main__":
    artifact_name = "syde-577/semantic-foggy-driving/UNet_ep_3_img_128x256_bs_4_channels_32_mIoU_0.41:latest"  # Replace with your artifact name
    wandb.init(project="semantic-foggy-driving", job_type="test")

    # Load model and configuration
    model, config = load_model_and_config(artifact_name, UNet, device)
    wandb.config.update(config)
    config = wandb.config

    # Load the test dataset
    test_data = get_data(config, split="test")
    test_loader = make_dataloader(test_data, batch_size=1, shuffle=False)

    # Evaluate and log the test mIoU
    test_mIoU = evaluate_model(model, test_loader)
    print(f"Test mIoU: {test_mIoU}")

    # Log the result to W&B
    wandb.log({"test_mIoU": test_mIoU})
