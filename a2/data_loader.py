import csv
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch
import random


class SimpleDataset(Dataset):
    """SimpleDataset [summary]
    
    [extended_summary]
    
    :param path_to_csv: [description]
    :type path_to_csv: [type]
    """
    def __init__(self, path_to_csv, transform=None):
        ## TODO: Add code to read csv and load data. 
        ## You should store the data in a field.
        # Eg (on how to read .csv files):
        # with open('path/to/.csv', 'r') as f:
        #   lines = ...
        ## Look up how to read .csv files using Python. This is common for datasets in projects.
        inp_df = pd.read_csv(path_to_csv)
        self.data = inp_df.to_numpy()
        self.num_features = self.data.shape[1]-1
        self.features = self.data[:, :-1]
        self.labels = self.data[:, -1]
        self.transform = transform
        

    def __len__(self):
        """__len__ [summary]
        
        [extended_summary]
        """
        ## TODO: Returns the length of the dataset.
        return self.data.shape[0]

    def __getitem__(self, index):
        """__getitem__ [summary]
        
        [extended_summary]
        
        :param index: [description]
        :type index: [type]
        """
        ## TODO: This returns only ONE sample from the dataset, for a given index.
        ## The returned sample should be a tuple (x, y) where x is your input 
        ## vector and y is your label
        ## Before returning your sample, you should check if there is a transform
        ## specified, and apply that transform to your sample
        # Eg:
        # if self.transform:
        #   sample = self.transform(sample)
        ## Remember to convert the x and y into torch tensors.

        sample = self.data[index,:]
        if self.transform:
            sample = self.transform(sample)
        # make tensor below -E
        x = torch.from_numpy(np.array(sample[:-1]))
        y = torch.from_numpy(np.array(sample[-1]))
        return x,y


def get_data_loaders(path_to_csv, 
                     transform_fn=None,
                     train_val_test=[0.8, 0.1, 0.1], 
                     batch_size=32):
    """get_data_loaders [summary]
    
    [extended_summary]
    
    :param path_to_csv: [description]
    :type path_to_csv: [type]
    :param train_val_test: [description], defaults to [0.8, 0.2, 0.2]
    :type train_val_test: list, optional
    :param batch_size: [description], defaults to 32
    :type batch_size: int, optional
    :return: [description]
    :rtype: [type]
    """
    # First we create the dataset given the path to the .csv file
    dataset = SimpleDataset(path_to_csv, transform=transform_fn)

    # Then, we create a list of indices for all samples in the dataset.
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)

    ## TODO: Rewrite this section so that the indices for each dataset split
    ## are formed.

    ## BEGIN: YOUR CODE
    train_indices = indices[:int(dataset_size*train_val_test[0])]
    val_indices = indices[int(dataset_size*train_val_test[0]):int(dataset_size*(train_val_test[0]+train_val_test[1]))]
    test_indices = indices[int(dataset_size*(train_val_test[0]+train_val_test[1])):]
    ## END: YOUR CODE

    # Now, we define samplers for each of the train, val and test data
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

    val_sampler = SubsetRandomSampler(val_indices)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader


def get_all_sample_features(path_to_csv):
    dataset = SimpleDataset(path_to_csv=path_to_csv)
    return dataset.features


def get_all_sample_labels(path_to_csv):
    dataset = SimpleDataset(path_to_csv=path_to_csv)
    return dataset.labels