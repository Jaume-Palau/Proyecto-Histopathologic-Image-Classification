import torch
import pandas as pd
from pathlib import Path
from PIL import Image
import sys, os

from config import *
from torchvision.transforms import transforms, v2


class DatasetWrapper_Train(torch.utils.data.Dataset):
    # TODO: Redo the function so that data can be downloaded without being on Kaggle
    
    def __init__(self):
        """PyTorch utility wrapper that provides a consistent interface for our data."""
                
        ## Construct an image transformer
        to_tensor_transformer = transforms.ToTensor()
        center_crop_transformer = v2.CenterCrop(size=(46, 46))
        composite_transformer = transforms.Compose([to_tensor_transformer,  
                                                    center_crop_transformer,
                                                   ])
        self.transformer = composite_transformer
        
        # Gather training data
        cwd = Path.cwd()
        input_path = Path(DATA_DIR)
        dataset_name = "histopathologic-cancer-detection"
        data_dir = Path(TRAIN_DIR)
        self.list_of_filenames = [file.name for file in data_dir.iterdir()]
        self.list_of_fullpaths = list(data_dir.iterdir())
        self.dataset_size = len(self.list_of_fullpaths)
        
        # Gather labels
        labels_path = Path(TRAIN_LABELS_CSV)
        labels_df = pd.read_csv(labels_path).set_index("id")
        ## Matching the label with each file name
        self.labels = [labels_df.loc[name[:-4], "label"] for name in self.list_of_filenames]
        self.labels_count = len(self.labels)
        
        
    def __len__(self):
        """Returns the number of data entries."""
        return self.dataset_size
    
    def __getitem__(self, idx): 
        """Get the i-th entry of transformed data.
        
        Args: 
            idx (int): Index of the image and label to get.
        
        Returns:
            (img, label): Transformed image as PyTorch Tensor, label as an int.
        """
        # Src: https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html#sphx-glr-auto-examples-transforms-plot-transforms-illustrations-py
        with Image.open(self.list_of_fullpaths[idx]) as image: 
            image_transformed = self.transformer(image)
            label = self.labels[idx]
        return (image_transformed, label)
    
    def get_untransformed(self, idx): 
        """Get the i-th entry of untransformed data.
        
        Args: 
            idx (int): Index of the image and label to get.
        
        Returns:
            (img, label): Untransformed image as a PIL image, label as an int.
        """
        image = Image.open(self.list_of_fullpaths[idx])
        label = self.labels[idx]
        
        return (image, label)

    
class DatasetWrapper_Test(torch.utils.data.Dataset):
    # TODO: Redo the function so that data can be downloaded without being on Kaggle
    
    def __init__(self):
        """PyTorch utility wrapper that provides a consistent interface for our data."""
                
        ## Construct an image transformer
        to_tensor_transformer = transforms.ToTensor()
        center_crop_transformer = v2.CenterCrop(size=(46, 46))
        composite_transformer = transforms.Compose([to_tensor_transformer,  
                                                    center_crop_transformer,
                                                   ])
        self.transformer = composite_transformer
        
        # Gather training data
        cwd = Path.cwd()
        input_path = Path(DATA_DIR)
        dataset_name = "histopathologic-cancer-detection"
        data_dir = Path(TEST_DIR)
        self.list_of_filenames = [file.name for file in data_dir.iterdir()]
#         self.list_of_fullpaths = list(data_dir.iterdir())
        self.list_of_fullpaths = [Path(data_dir, filename) for filename in self.list_of_filenames]
        self.dataset_size = len(self.list_of_fullpaths)
        
        
    def __len__(self):
        """Returns the number of data entries."""
        return self.dataset_size
    
    def __getitem__(self, idx): 
        """Get the i-th entry of transformed data.
        
        Args: 
            idx (int): Index of the image and label to get.
        
        Returns:
            img: Transformed image as PyTorch Tensor
        """
        # Src: https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html#sphx-glr-auto-examples-transforms-plot-transforms-illustrations-py
        with Image.open(self.list_of_fullpaths[idx]) as image: 
            image_transformed = self.transformer(image)
        file_id = self.list_of_filenames[idx][:-4]
        return (file_id, image_transformed)
    
    def get_untransformed(self, idx): 
        """Get the i-th entry of untransformed data.
        
        Args: 
            idx (int): Index of the image and label to get.
        
        Returns:
            (img, label): Untransformed image as a PIL image, label as an int.
        """
        image = Image.open(self.list_of_fullpaths[idx])
        file_id = self.list_of_filenames[idx][:-4]
        
        return (file_id, image)


################################################################################
## Train Test Split
################################################################################

def train_test_split(dataset_to_split:torch.utils.data.Dataset, 
                     train_proportion:float=0.8, 
                     random_seed:int=RANDOM_SEED, 
                     subset_size:int=None):
    """Split a dataset to train and validation sets.
    
    Args: 
        dataset_to_split (torch.utils.data.Dataset): The dataset to be split.
        train_proportion (float): Proportion of training set, default=0.8.
        random_seed (int): Seed for random generator for reproducibility.
        subset_size (int): If set, the dataset to be split will be truncated to this size.
    
    Returns: 
        (train_set, test_set)
    """
    assert ((train_proportion>0) & (train_proportion<1)), f"Train proportion has to be (0, 1), got {train_proportion}."
    
    if subset_size != None:
        dataset_to_split = torch.utils.data.Subset(dataset_to_split, range(subset_size))
    
    generator1 = torch.Generator().manual_seed(1234)  # To allow reproducibility

    # Split to train/test subsets (including labels)
    train_set, test_set = torch.utils.data.random_split(
        dataset_to_split, 
        [train_proportion, 1-train_proportion], 
        generator=generator1
    )
    
    return (train_set, test_set)



if __name__ == "__main__":
    ## Gather and transform the image
    train_dataset = DatasetWrapper_Train()
    print(f"training_data_tensor size: {sys.getsizeof(train_dataset)} bytes")


    test_dataset = DatasetWrapper_Test()
    print(f"testing_data_tensor size: {sys.getsizeof(test_dataset)} bytes")

    ## IGNORE - sanity check
    training_set, testing_set = train_test_split(train_dataset)

    print(f"The training set has {len(training_set)} entries.")
    print(f"The testing set has {len(testing_set)} entries.")
    print()

    for i, (image, label) in enumerate(training_set): 
        if i > 10: break  # Checking the first few entries
        print(f"Image: {i} | On: {image.device} | Shape: {image.shape} | Label: {label}")