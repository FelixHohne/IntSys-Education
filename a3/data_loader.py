import csv
import os
import pickle
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


class SimpleDataset(Dataset):
    """SimpleDataset [summary]
    
    [extended_summary]
    
    :param path_to_pkl: Path to PKL file with Images
    :type path_to_pkl: str
    :param path_to_labels: path to file with labels
    :type path_to_labels: str
    """
    def __init__(self, path_to_pkl, path_to_labels):
        ## TODO: Add code to read csv and load data. 
        ## You should store the data in a field.
        # Eg (on how to read .csv files):
        # with open('path/to/.csv', 'r') as f:
        #   lines = ...
        ## Look up how to read .csv files using Python. This is common for datasets in projects.
        pkl_file = open(path_to_pkl, 'rb')
        self.data = pickle.load(pkl_file)
        pkl_file.close()
        self.labels = np.fromfile(path_to_labels)

    def __len__(self):
        """__len__ [summary]
        [extended_summary]
        """
        return self.labels.shape[0] 

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
        tup_item = (torch.tensor(self.data[index]), torch.tensor(self.labels[index]))
        return tup_item


def get_data_loaders(path_to_pkl, 
                     path_to_labels,
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
    dataset = SimpleDataset(path_to_pkl, path_to_labels)

    # Then, we create a list of indices for all samples in the dataset.
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)

    ## TODO: Rewrite this section so that the indices for each dataset split
    ## are formed. You can take your code from last time

    ## BEGIN: YOUR CODE
    t_inds = train_val_test[0]*len(dataset)
    v_inds = t_inds + train_val_test[1]*len(dataset)
    train_indices = indices[:t_inds]
    val_indices = indices[t_inds:v_inds]
    test_indices = indices[v_inds:]
    ## END: YOUR CODE

    # Now, we define samplers for each of the train, val and test data
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

    val_sampler = SubsetRandomSampler(val_indices)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader