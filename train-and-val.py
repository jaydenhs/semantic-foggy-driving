import numpy as np
import torch
import wandb
import random
from tqdm.auto import tqdm
import torch.nn as nn
from sklearn.metrics import jaccard_score
from torch.utils.data import Subset
from torch.optim.lr_scheduler import StepLR

from models.unet import UNet
from dataset import get_data, make_dataloader
from labels import map_trainId_to_label_name
import os

config = dict(
    architecture="UNet",
    base_channels = 32,
    batch_size = 4,
    epochs = 10,
    image_size = (512, 1024),
    input_beta = 0.01,
    lr = 0.001,
    lr_step_size = 2,
    lr_gamma = 0.1,
    log_freq = 8,
    root_path = r".\data_split"
)

# Set random seeds
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)
torch.cuda.empty_cache() 

wandb.login()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
def model_pipeline(config):
    run_name = f"{config['architecture']}_ep_{config['epochs']}_img_{int(config['image_size'][0])}x{int(config['image_size'][1])}_bs_{config['batch_size']}_channels_{config['base_channels']}"

    with wandb.init(group="syde-577", name=run_name, project="semantic-foggy-driving", config=config):
        config = wandb.config

        wandb.define_metric("train/batch_num")
        wandb.define_metric("train/*", step_metric="train/batch_num")

        wandb.define_metric("epoch")
        wandb.define_metric("val/*", step_metric="epoch")

        model, train_loader, val_loader, criterion, optimizer, scheduler = make(config)

        best_val_mIoU = float('-inf')
        best_model_state = None
        for epoch in range(config.epochs):
            train(model, train_loader, criterion, optimizer, scheduler, config, epoch)
            val_mIoU = validate(model, val_loader, criterion, config, epoch)

            if val_mIoU > best_val_mIoU:
                best_val_mIoU = val_mIoU
                print(f"New best model found with val mIoU: {best_val_mIoU}")
                best_model_state = model.state_dict()
        
        # Save best model as W&B artifact
        if best_model_state is not None:
            artifact_name = f"{run_name}_mIoU_{best_val_mIoU:.2f}"
            model_artifact = wandb.Artifact(artifact_name, type="model")
            model_artifact.metadata = dict(config)

            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            model_path = f"checkpoints/{artifact_name}.pth"

            torch.save(best_model_state, model_path)
            model_artifact.add_file(model_path)
            wandb.log_artifact(model_artifact)
            
    return

def make(config):
    train, val = get_data(config, split="train"), get_data(config, split="val")
    
    # For quick debugging
    train_subset = Subset(train, range(len(train)))
    val_subset = Subset(val, range(len(val)))
    
    train_loader = make_dataloader(train_subset, batch_size=config.batch_size, shuffle=True)
    val_loader = make_dataloader(val_subset, batch_size=config.batch_size, shuffle=False)

    model = UNet(in_channels=3, num_classes=19, base_channels=config.base_channels).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr)
    
    scheduler = StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)

    return model, train_loader, val_loader, criterion, optimizer, scheduler

def train(model, loader, criterion, optimizer, scheduler, config, epoch):
    model.train()
    wandb.watch(model, criterion, log="all", log_freq=10)

    example_ct = 0 
    total_loss = 0
    total_mIoU = 0
    pbar = tqdm(loader, desc=f"Training: Epoch {epoch + 1}/{config.epochs}", leave=True)
    for batch, sample in enumerate(pbar):
        inputs = sample['input_image']
        input_names = sample['input_name']
        labels = sample['target_image']

        inputs = inputs.permute(0, 3, 1, 2).to(device)
        labels = labels.to(device)

        optimizer.zero_grad() 

        outputs = model(inputs)  
        loss = criterion(outputs, labels)  
        loss.backward() 

        optimizer.step() 

        total_loss += loss.item()
        example_ct += len(inputs)

        mIoU = calculate_mIoU(outputs, labels)
        pbar.set_postfix(loss=loss.item(), mIoU=mIoU)
        total_mIoU += mIoU

        train_batch_num = epoch * len(loader) + batch
        if batch % config.log_freq == 0:
            wandb.log({
                "train/batch_num": train_batch_num,
                "train/loss": loss.item(),
                "train/mIoU": mIoU,
                "train/learning_rate": scheduler.get_last_lr()[0]
            })
            log_sample(inputs[0], input_names[0], labels[0], outputs[0], split='train')
    
    scheduler.step()

def validate(model, loader, criterion, config, epoch):
    model.eval()
    total_loss = 0
    total_mIoU = 0
    example_ct = 0 

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Validation: Epoch {epoch + 1}/{config.epochs}", leave=True)
        for batch, sample in enumerate(pbar):
            inputs = sample['input_image']
            input_names = sample['input_name']
            labels = sample['target_image']

            inputs = inputs.permute(0, 3, 1, 2).to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            example_ct += len(inputs)

            mIoU = calculate_mIoU(outputs, labels)
            total_mIoU += mIoU
            
            pbar.set_postfix(loss=loss.item(), mIoU=mIoU)

            if batch % config.log_freq == 0:
                log_sample(inputs[0], input_names[0], labels[0], outputs[0], split='val')

    avg_loss = total_loss / len(loader)
    avg_mIoU = total_mIoU / len(loader)

    wandb.log({
        "epoch": epoch,
        "val/loss": avg_loss,
        "val/mIoU": avg_mIoU
    })

    return avg_mIoU

def calculate_mIoU(outputs, labels):
    outputs = torch.argmax(outputs, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    
    mask = labels != 255
    
    outputs = outputs[mask]
    labels = labels[mask]
    
    mIoU = jaccard_score(labels.flatten(), outputs.flatten(), average='weighted')
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

if __name__ == "__main__":
    model_pipeline(config)
