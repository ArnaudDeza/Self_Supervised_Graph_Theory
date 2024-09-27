from typing import Optional, Tuple

import torch
import pickle
import random
import numpy as np
from torch.utils.data import Dataset



class RandomGraphDataset(Dataset):
    """
    Dataset of graphs for self-supervised learning in search for counterexamples of graph theory conjectures.

    Each problem instance is a dictionary of the form
    
    
    """

    def __init__(self, 
                model_type: str,
                expert_pickle_file: str,
                batch_size: int,
                data_augmentation: bool = False,
                data_augmentation_linear_scale: bool = False,
                augment_direction: bool = False,
                custom_num_instances: Optional[int] = None,
                custom_num_batches: Tuple[str, int] = None
                ):
        
        """
        Parameters:
            model_type [str]: Either "BQ" or "LEHD"
            expert_pickle_file [str]: Path to file with expert trajectories.
            batch_size [int]: Number of items to return in __getitem__
            data_augmentation [bool]: If True, a geometric augmentation is performed on the instance
                before choosing subtour.
            data_augmentation_linear_scale [bool]: If True, linear scaling is performed in geometric augmentation.
                Not that this changes the distribution of the points.
            augment_direction [bool]: If True, direction of the subtour is randomly swapped.
            custom_num_instances [int]: If given, only the first num instances are taken.
            custom_num_batches [Tuple[str, int]]: If the first entry is "multiplier", the number of batches (i.e. length of dataset)
                is equal to the given value multiplied with the number of instances in the dataset.
                If the first entry is "absolute" the given value is explicitly taken as the number of samples.
        """


        self.model_type = model_type
        self.expert_pickle_file = expert_pickle_file
        self.data_augmentation = data_augmentation
        self.data_augmentation_linear_scale = data_augmentation_linear_scale
        self.augment_direction = augment_direction
        self.batch_size = batch_size
        with open(expert_pickle_file, "rb") as f:
            self.instances = pickle.load(f)

        if custom_num_instances is not None:
            self.instances = self.instances[:custom_num_instances]

        print(f"Loaded dataset. Num items: {len(self.instances)}")

        self.num_nodes = self.instances[0]["inst"].shape[0]
        # One instance corresponds to one random subtour, so
        # length of dataset corresponds to the length of one epoch.
        if custom_num_batches is None:
            self.length = len(self.instances) // self.batch_size
        elif custom_num_batches[0] == "absolute":
            self.length = custom_num_batches[1]
        elif custom_num_batches[0] == "multiplier":
            self.length = custom_num_batches[1] * len(self.instances)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # TODO: Implement this method
        pass