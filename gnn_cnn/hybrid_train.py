##################################################
# 1. Imports
##################################################
import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader, Subset
from torch_geometric.data import Dataset as GeometricDataset, DataLoader as GeometricDataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree

# If you have custom modules like DeePore
from . import DeePore as dp

##################################################
# 2. Dataset Classes
##################################################
class PoreNetworkDataset(GeometricDataset):
    """
    GNN dataset. It loads graph files from a directory and corresponding labels.
    """
    def __init__(self, graphs_dir, labels_file, num_samples=None, transform=None, pre_transform=None):
        super().__init__(graphs_dir, transform, pre_transform)
        self.graphs_dir = graphs_dir
        self.labels = self.load_labels(labels_file)
        self.num_samples = num_samples if num_samples else len(self.labels)

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


class PermeabilityDataset(TorchDataset):
    """
    CNN dataset. It loads 2D slices/images plus corresponding ground-truth values (e.g., permeability).
    """
    def __init__(self, file_path, start_index, end_index):
        self.file_path = file_path
        self.start_index = start_index
        self.end_index = end_index
        self.images = []
        self.ground_truths = []

        for index in range(self.start_index, self.end_index + 1):
            im = np.squeeze(dp.readh5slice(self.file_path, 'X', [index]))
            props = dp.readh5slice(self.file_path, 'Y', [index])
            gt = props[0][0]
            self.images.append(im)
            self.ground_truths.append(gt)
        
        self.images = np.array(self.images)
        self.ground_truths = np.array(self.ground_truths)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        ground_truth = self.ground_truths[idx]
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        ground_truth = torch.tensor(ground_truth, dtype=torch.float32)
        return image, ground_truth

##################################################
# 3. Data Preparation
##################################################
# Example default paths—adjust to your folder structure
gnn_num_samples = 5000  # Number of samples in the GNN dataset
graphs_dir = 'data/data_inverse/graphs'
labels_file = 'data/data_inverse/test_GT.pkl'

file_path = 'data/DeePore_Compact_Data.h5'
start_index = 0
end_index = 4999  # Must match gnn_num_samples if they're supposed to have the same length

# Create GNN and CNN datasets
gnn_dataset = PoreNetworkDataset(graphs_dir=graphs_dir,
                                 labels_file=labels_file,
                                 num_samples=gnn_num_samples)

cnn_dataset = PermeabilityDataset(file_path=file_path,
                                  start_index=start_index,
                                  end_index=end_index)

# Ensure they have the same length
assert len(gnn_dataset) == len(cnn_dataset), (
    f"Datasets should have the same length: GNN length={len(gnn_dataset)}, "
    f"CNN length={len(cnn_dataset)}"
)

# Shuffle and create split indices
indices = list(range(len(gnn_dataset)))
random.shuffle(indices)
train_size = int(0.8 * len(indices))
test_size = len(indices) - train_size
train_indices = indices[:train_size]
test_indices = indices[train_size:]

# Create subsets
gnn_train_dataset = Subset(gnn_dataset, train_indices)
gnn_test_dataset = Subset(gnn_dataset, test_indices)
cnn_train_dataset = Subset(cnn_dataset, train_indices)
cnn_test_dataset = Subset(cnn_dataset, test_indices)

# DataLoaders
batch_size = 16
gnn_train_loader = GeometricDataLoader(gnn_train_dataset, batch_size=batch_size, shuffle=True)
gnn_test_loader  = GeometricDataLoader(gnn_test_dataset,  batch_size=batch_size, shuffle=False)
cnn_train_loader = TorchDataLoader(cnn_train_dataset,     batch_size=batch_size, shuffle=True)
cnn_test_loader  = TorchDataLoader(cnn_test_dataset,      batch_size=batch_size, shuffle=False)

# Determine node/edge feature dimensions from the first sample
sample_gnn_data = gnn_dataset[0]
node_in_feats = sample_gnn_data.x.shape[1]
edge_in_feats = sample_gnn_data.edge_attr.shape[1]

##################################################
# 4. Model Definitions
##################################################
class GNNLayer(MessagePassing):
    """
    A basic GNN layer with message passing.
    """
    def __init__(self, node_dims, edge_dims, output_dims):
        super().__init__(aggr='add')  # "Add" aggregation.
        self.W_msg = nn.Linear(node_dims + edge_dims, node_dims)
        self.W_apply = nn.Linear(node_dims * 2, output_dims)
        
    def forward(self, x, edge_index, edge_attr):
        # Add self-loops
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value="mean")

        # Compute node degrees
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        norm = deg.pow(-0.5)
        norm[deg == 0] = 0
        edge_weight = norm[row] * norm[col]

        # Start propagating messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_weight=edge_weight)
        return out

    def message(self, x_j, edge_attr, edge_weight):
        msg = torch.cat([x_j, edge_attr], dim=1)
        return edge_weight.view(-1, 1) * F.relu(self.W_msg(msg))

    def update(self, aggr_out, x):
        aggr_out = torch.cat([x, aggr_out], dim=1)
        return F.relu(self.W_apply(aggr_out))


class GCN(nn.Module):
    """
    2-layer GCN (with an optional second layer, commented out).
    """
    def __init__(self, node_feats, edge_feats, hidden_feats1, hidden_feats2, predictor_hidden_feats=128):
        super().__init__()
        self.conv1 = GNNLayer(node_feats, edge_feats, hidden_feats1)
        self.conv2 = GNNLayer(hidden_feats1, edge_feats, hidden_feats2)
        self.readout = global_mean_pool
        self.fc1_ln = nn.LayerNorm(hidden_feats2)  # using LayerNorm
        self.predict1 = nn.Linear(hidden_feats2, predictor_hidden_feats)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        
        if data.batch is not None:
            batch = data.batch.to(device)
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long).to(device)

        h = self.conv1(x, edge_index, edge_attr)
        # h = self.conv2(h, edge_index, edge_attr)  # uncomment if you want a second layer
        graph_feats = self.readout(h, batch)
        graph_feats = self.fc1_ln(graph_feats)
        graph_feats = F.relu(self.predict1(graph_feats))
        return graph_feats


class CNNModel(nn.Module):
    """
    A simple CNN for 2D images with 3 channels.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(12, 24, kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        
        # Adjust fc1_input_dim based on image size + pooling
        self.fc1_input_dim = 24 * 16 * 16  
        self.fc1 = nn.Linear(self.fc1_input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


class HybridModel(nn.Module):
    """
    Combines a GCN (for graph input) and a CNN (for image input).
    """
    def __init__(self, gnn_output_dim, cnn_output_dim, hidden_dim):
        super().__init__()
        # Instantiate GCN + CNN with known input dims
        self.gnn = GCN(node_in_feats, edge_in_feats, 256, 256, 128)
        self.cnn = CNNModel()

        # MLP on top of concatenated features
        self.fc1 = nn.Linear(gnn_output_dim + cnn_output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, graph_data, image_data):
        gnn_features = self.gnn(graph_data)
        cnn_features = self.cnn(image_data)
        
        combined_features = torch.cat((gnn_features, cnn_features), dim=1)
        x = F.relu(self.fc1(combined_features))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

##################################################
# 5. Training and Testing Functions
##################################################
def train(
    model,
    gnn_loader,
    cnn_loader,
    optimizer,
    criterion,
    epoch,
    num_epochs,
    train_preds_list,
    train_labels_list,
    batch_size
):
    model.train()
    total_loss = 0
    for gnn_data, (cnn_images, cnn_labels) in zip(gnn_loader, cnn_loader):
        gnn_data = gnn_data.to(device)
        cnn_images = cnn_images.to(device)
        cnn_labels = cnn_labels.to(device).unsqueeze(1)
        cnn_labels = cnn_labels.squeeze(2)  # reshape if needed
        
        optimizer.zero_grad()
        output = model(gnn_data, cnn_images)
        loss = criterion(output, cnn_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Collect predictions at last epoch
        if epoch == num_epochs - 1:
            if output.ndim == 1:
                train_preds_list.append(output.item())
                train_labels_list.append(cnn_labels.item())
            else:
                for b in range(len(output)):
                    train_preds_list.append(output[b].item())
                    train_labels_list.append(cnn_labels[b].item())
    
    return total_loss / len(gnn_loader)


def test(
    model,
    gnn_loader,
    cnn_loader,
    criterion,
    epoch,
    num_epochs,
    test_preds_list,
    test_labels_list,
    batch_size
):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for gnn_data, (cnn_images, cnn_labels) in zip(gnn_loader, cnn_loader):
            gnn_data = gnn_data.to(device)
            cnn_images = cnn_images.to(device)
            cnn_labels = cnn_labels.to(device).unsqueeze(1)
            cnn_labels = cnn_labels.squeeze(2)

            output = model(gnn_data, cnn_images)
            loss = criterion(output, cnn_labels)
            total_loss += loss.item()

            # Collect predictions at last epoch
            if epoch == num_epochs - 1:
                if output.ndim == 1:
                    test_preds_list.append(output.item())
                    test_labels_list.append(cnn_labels.item())
                else:
                    for b in range(len(output)):
                        test_preds_list.append(output[b].item())
                        test_labels_list.append(cnn_labels[b].item())

    return total_loss / len(gnn_loader)

##################################################
# 6. Main Code: Hyperparameters, Training Loop, etc.
##################################################
if __name__ == "__main__":
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Hyperparameters
    learning_rate = 0.0001
    num_epochs = 40
    criterion = nn.MSELoss()

    # Set up the model
    gnn_output_dim = 128
    cnn_output_dim = 64
    hidden_dim = 128

    model = HybridModel(gnn_output_dim, cnn_output_dim, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Lists for tracking performance
    train_losses = []
    test_losses = []
    train_preds_all = []
    train_labels_all = []
    test_preds_all = []
    test_labels_all = []

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train(
            model,
            gnn_train_loader,
            cnn_train_loader,
            optimizer,
            criterion,
            epoch,
            num_epochs,
            train_preds_all,
            train_labels_all,
            batch_size
        )
        test_loss = test(
            model,
            gnn_test_loader,
            cnn_test_loader,
            criterion,
            epoch,
            num_epochs,
            test_preds_all,
            test_labels_all,
            batch_size
        )
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        print(f"Epoch: {epoch+1:2d}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    # Convert predictions and labels to numpy arrays for plotting
    train_preds_all = np.array(train_preds_all)
    train_labels_all = np.array(train_labels_all)
    test_preds_all = np.array(test_preds_all)
    test_labels_all = np.array(test_labels_all)

    # Plot: Training and Test Losses
    plt.figure(figsize=(8, 4))
    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.plot(range(num_epochs), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Losses')
    plt.show()

    # Plot: Test Predictions vs. Ground Truth
    plt.figure(figsize=(8, 4))
    plt.scatter(test_labels_all, test_preds_all, alpha=0.5)
    min_val, max_val = test_labels_all.min(), test_labels_all.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.title('Test Predictions vs Ground Truth')
    plt.show()

    # Plot: Train Predictions vs. Ground Truth
    plt.figure(figsize=(8, 4))
    plt.scatter(train_labels_all, train_preds_all, alpha=0.5)
    min_val, max_val = train_labels_all.min(), train_labels_all.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.title('Train Predictions vs Ground Truth')
    plt.show()
