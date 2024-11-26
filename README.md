# Semantic Foggy Scene Understanding with Synthetic Data

Building upon previous research: https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/.

We investigate training purely on synthetic data (no pre-training), comparing a CNN and Vision Transformer with a baseline UNet.

Steps to set-up:

1. Download gtFine_trainvaltest and leftImg8bit_trainvaltest_foggy from https://www.cityscapes-dataset.com/ (need to create an account)
2. Place them both into data/
3. Run split-dataset.py to generate a data_split/ (establishes train/val/test splits for fully-annotated data)

Use `train-and-val.py` to train a model and save the model state which produces the best validation mIoU into checkpoints/ and onto wandb.

`test.py` loads the checkpoint from wandb and evaluates its mIoU on the test set.
