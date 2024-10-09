import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import random
from skimage import io, transform
from labels import map_id_to_trainId
import wandb
from models.unet import UNet

from tqdm.auto import tqdm

import torch.nn as nn
import torch.nn.functional as F

class YourCustomDataset(torch.utils.data.Dataset):
    def __init__(self, input, target, split, config):
        """
        Args:
            root_path (str): path to data
            input (str): specific path to input data 
            target (str): specific path to target (annotation) data
            split (str): split to get data from ("train", "test", or "val")
            input_beta (float): defines the synthetic fog density to filter the images to
            train_transform (bool, optional): Applies the defined data transformations used in training. Defaults to None.
        """
        super(YourCustomDataset, self).__init__()
        self.root_path = config.root_path
        self.input = input
        self.target = target
        self.split = split
        self.input_beta = config.input_beta
        self.image_size = config.image_size

        # iterates through split of data and creates an array of image names filtered to specified fog beta
        self.image_names = []
        X_SPLIT_PATH = os.path.join(self.root_path, self.input, self.split)
        for CITY_NAME in os.listdir(X_SPLIT_PATH):
            CITY_PATH = os.path.join(X_SPLIT_PATH, CITY_NAME)
            for image_name in os.listdir(CITY_PATH):
                if str(self.input_beta) in image_name:
                    IMAGE_PATH = os.path.join(CITY_PATH, image_name)
                    self.image_names.append(IMAGE_PATH)

        # same for annotation_names, filters to label annotation_names
        self.annotation_names = []
        Y_SPLIT_PATH = os.path.join(self.root_path, self.target, self.split)
        for CITY_NAME in os.listdir(Y_SPLIT_PATH):
            CITY_PATH = os.path.join(Y_SPLIT_PATH, CITY_NAME)
            for annotation_name in os.listdir(CITY_PATH):
                if "label" in annotation_name:
                    ANNOTATION_PATH = os.path.join(CITY_PATH, annotation_name)
                    self.annotation_names.append(ANNOTATION_PATH)

    def __len__(self):
        number_files_input = len(self.image_names)
        number_files_target = len(self.annotation_names)

        if number_files_input == number_files_target:
            return number_files_input
        else:
            return f"Input: {number_files_input} does not match Target: {number_files_target}"
        
    def __getitem__(self, idx):
        """
        Necessary function that loads and returns a sample from the dataset at a given index. 
        https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
        
        Based on the index,it identifies the input and target images location on the disk, 
        reads both items as a numpy array (float32). If the train_transform argument is True, 
        the above defined train transformations are applied. Else, the test transformations are applied
        Args:
            idx (iterable): 
        Returns:
            tensors: input and target image
        """
        input_path = self.image_names[idx]
        target_path = self.annotation_names[idx]

        # Read images
        input_image = io.imread(input_path)
        target_image = io.imread(target_path)

        # Resize images
        size = self.image_size
        input_image = transform.resize(input_image, (size, size), order=0)
        target_image = transform.resize(target_image, (size, size), order=0)

        # Map target IDs to training IDs
        target_image = map_id_to_trainId(target_image)

        # Convert to PyTorch tensors
        input_image = torch.from_numpy(input_image).float() / 255.0
        target_image = torch.from_numpy(target_image).long()

        # Create the sample dictionary
        sample = {
            'input_image': input_image,
            'input_name': input_path.split("\\")[-1],
            'target_image': target_image,
            'target_name': target_path.split("\\")[-1]
        }


        return sample    
    
def plot_sample(sample):
    plt.figure(figsize=(16,8))
    plt.subplot(121)
    plt.title(sample['input_name'])
    plt.imshow(sample['input_image'])
    plt.subplot(122)
    plt.title(f"{sample['target_name']}")
    plt.imshow(sample['target_image'],vmin=0, vmax=20)
    plt.colorbar()
    plt.show()

torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)
torch.cuda.empty_cache()  # Clear cached memory

wandb.login()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

config = dict(
    architecture="UNet",
    batch_size = 4,
    epochs = 1,
    image_size = 128,
    input_beta = 0.01,
    learning_rate = 0.001,
    root_path = r"C:\Users\Jayden Hsiao\Documents\Grad School\02 - Fall 2024\SYDE 577\semantic-foggy-driving\data"
)
    
def model_pipeline(config):
    run_name = f"{config['architecture']}_ep_{config['epochs']}_img_{config['image_size']}_bs_{config['batch_size']}"

    # tell wandb to get started
    with wandb.init(group="syde-577", name=run_name, project="semantic-foggy-driving", config=config):
        # access all config through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        model, train_loader, test_loader, criterion, optimizer = make(config)

        # and use them to train the model
        train(model, train_loader, criterion, optimizer, config)

        # # and test its final performance
        test(model, test_loader, criterion, config)

    return

def make(config):
    # Make the data
    train, test = get_data(config, split="train"), get_data(config, split="val")
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # plot_sample(train[0])

    # Make the model
    model = UNet(in_channels=3, num_classes=19).to(device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)

    return model, train_loader, test_loader, criterion, optimizer

def get_data(config, split):
    INPUT_PATH = r"leftImg8bit_trainvaltest_foggy\leftImg8bit_foggy"
    TARGET_PATH = r"gtFine_trainvaltest\gtFine"

    dataset = YourCustomDataset(input=INPUT_PATH, target=TARGET_PATH, split=split, config=config)
    
    return dataset

def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True, pin_memory=True)
    return loader

def train(model, loader, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in tqdm(range(config.epochs), desc="Epochs"):
        print(f"--- EPOCH {epoch} ---")
        for _, sample in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}/{config.epochs}", leave=False)):
            inputs = sample['input_image']
            labels = sample['target_image']

            loss, outputs = train_batch(inputs, labels, model, optimizer, criterion)
            example_ct += len(inputs)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct - 1) % 25) == 0:
                wandb.log({"train_loss": loss}, step=example_ct)
                # log_sample(inputs[0], labels[0], outputs[0])


def train_batch(inputs, labels, model, optimizer, criterion):
    # Move inputs and labels to the GPU
    # Permute input to [B, C, W, H]
    inputs = inputs.permute(0, 3, 1, 2).to(device)
    labels = labels.to(device)

    optimizer.zero_grad()  # Zero your gradients for every batch!

    outputs = model(inputs)  # Make predictions
    loss = criterion(outputs, labels)  # Compute the loss
    loss.backward()  # Backpropagate the gradients

    optimizer.step()  # Adjust learning weights

    return loss, outputs
    
def log_sample(input, label, output):
    # Detach and move to CPU
    input_img = input.detach().cpu().numpy()  # Convert to numpy for WandB
    label_img = label.detach().cpu().numpy()
    output_img = output.detach().cpu()
    
    # Convert the output to a single channel (class predictions)
    output_img = torch.argmax(output_img, dim=0).numpy()  # Change dim=1 to dim=0 for output_img shape

    # Log the first image to WandB
    wandb.log({
        "input_image": wandb.Image(input_img, caption="Input Image"),
        "label_image": wandb.Image(label_img, caption="Label Image"),
        "output_image": wandb.Image(output_img, caption="Output Image")
    })



def print_batch(inputs, labels, outputs):
    plt.figure(figsize=(12,4))
    input_img = inputs.detach().cpu()
    plt.subplot(131)
    plt.imshow(input_img[0, ...], vmin=0, vmax=20)

    label_img = labels.detach().cpu()
    plt.subplot(132)
    plt.imshow(label_img[0, ...], vmin=0, vmax=20)

    output_img = outputs.detach().cpu()
    output_img = torch.argmax(output_img, dim=1, keepdim=True)
    plt.subplot(133)
    plt.imshow(output_img[0, 0, ...], vmin=0, vmax=20)
    plt.show()

def test(model, loader, criterion, config):
    # Set the model to evaluation mode
    model.eval()

    total_loss = 0.0
    example_ct = 0 
    num_batches = 0

    with torch.no_grad():  # No need to compute gradients during testing
        for batch, sample in enumerate(tqdm(loader, desc="Testing", leave=False)):
            inputs = sample['input_image']
            labels = sample['target_image']

            # Move inputs and labels to the GPU
            inputs = inputs.permute(0, 3, 1, 2).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            example_ct += len(inputs)
            num_batches += 1

            # Log test metrics every 10 batches
            if batch % 10 == 0:
                wandb.log({"test_loss": loss}, step=example_ct)
                # log_sample(inputs[0], labels[0], outputs[0])

    # Compute the average loss over the validation set
    avg_loss = total_loss / num_batches
    print(f"Average Test Loss: {avg_loss:.3f}")
    wandb.log({"avg_test_loss": avg_loss})

    return avg_loss

# Call the main function to start the training
if __name__ == "__main__":
    model_pipeline(config)