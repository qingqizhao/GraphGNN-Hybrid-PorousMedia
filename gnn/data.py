import os
import torch
from torch_geometric.data import Dataset, DataLoader
import numpy as np
import pickle
from torch.utils.data import random_split, Subset

class PoreNetworkDataset(Dataset):
    def __init__(self, graphs_dir, labels_file, num_samples=6000, transform=None, pre_transform=None):
        super().__init__(graphs_dir, transform, pre_transform)
        self.graphs_dir = graphs_dir
        self.labels = self.load_labels(labels_file)
        self.num_samples = num_samples

    def load_labels(self, labels_file):
        with open(labels_file, 'rb') as f:
            labels = pickle.load(f)
        return labels

    def len(self):
        return self.num_samples

    def get(self, idx):
        graph_path = os.path.join(self.graphs_dir, f"graph_{idx}.pt")
        data = torch.load(graph_path)
        data.y = torch.tensor(self.labels[idx][0][0], dtype=torch.float)
        return data

def create_dataloaders(graphs_dir='data/data_inverse/graphs', labels_file='data/data_inverse/test_GT.pkl',
                       num_samples=6000, batch_size=32, train_split=0.8):
    """
    Creates train and test DataLoaders for the GNN.
    """
    dataset = PoreNetworkDataset(graphs_dir, labels_file, num_samples=num_samples)

    # Shuffle the dataset before splitting
    indices = torch.randperm(len(dataset)).tolist()
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, dataset
