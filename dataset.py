import torch
import os
from skimage import io, transform
from labels import map_id_to_trainId

class FoggyCityscapesDataset(torch.utils.data.Dataset):
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
        super(FoggyCityscapesDataset, self).__init__()
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
                if "labelId" in annotation_name:
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
        input_image = transform.resize(input_image, self.image_size, order=0)
        target_image = transform.resize(target_image, self.image_size, order=0)

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
    
def get_data(config, split):
    INPUT_PATH = r"leftImg8bit_trainvaltest_foggy\leftImg8bit_foggy"
    TARGET_PATH = r"gtFine_trainvaltest\gtFine"

    dataset = FoggyCityscapesDataset(input=INPUT_PATH, target=TARGET_PATH, split=split, config=config)
    
    return dataset

def make_dataloader(dataset, batch_size, shuffle):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=shuffle, pin_memory=True)
    return loader