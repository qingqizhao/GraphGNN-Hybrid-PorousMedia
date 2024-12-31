import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from gnn.data import create_dataloaders  
from gnn.model import GCN, device

# Training function
def train(model, loader, optimizer, criterion, epoch, num_epochs, train_preds_list, train_labels_list):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # If at the last epoch, store predictions for plotting
        if epoch == num_epochs - 1:
            if output.ndim == 1:
                train_preds_list.append(output.item())
                train_labels_list.append(data.y.item())
            else:
                for b in range(len(output)):
                    train_preds_list.append(output[b].item())
                    train_labels_list.append(data.y[b].item())
    
    return total_loss / len(loader)

# Testing function
def test(model, loader, criterion, epoch, num_epochs, test_preds_list, test_labels_list):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y.view(-1, 1))
            total_loss += loss.item()

            # If at the last epoch, store predictions for plotting
            if epoch == num_epochs - 1:
                if output.ndim == 1:
                    test_preds_list.append(output.item())
                    test_labels_list.append(data.y.item())
                else:
                    for b in range(len(output)):
                        test_preds_list.append(output[b].item())
                        test_labels_list.append(data.y[b].item())

    return total_loss / len(loader)

def main():
    # Hyperparameters
    learning_rate = 0.0001
    num_epochs = 100
    batch_size = 16
    criterion = nn.MSELoss()

    # Create data loaders
    train_loader, test_loader, dataset = create_dataloaders(batch_size=batch_size, train_split=0.8)

    # Determine node/edge features
    node_in_feats = dataset[0].x.shape[1]
    edge_in_feats = dataset[0].edge_attr.shape[1]

    # Initialize model, optimizer
    model = GCN(node_in_feats, edge_in_feats).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Track losses and predictions
    train_losses, test_losses = [], []
    train_preds_all, train_labels_all = [], []
    test_preds_all, test_labels_all = [], []

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, epoch, num_epochs,
                           train_preds_all, train_labels_all)
        test_loss = test(model, test_loader, criterion, epoch, num_epochs,
                         test_preds_all, test_labels_all)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        print(f'Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    # Convert predictions and labels to numpy arrays for plotting
    train_preds_all = np.array(train_preds_all)
    train_labels_all = np.array(train_labels_all)
    test_preds_all = np.array(test_preds_all)
    test_labels_all = np.array(test_labels_all)

    # Plot training and test losses
    plt.figure(figsize=(8, 4))
    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.plot(range(num_epochs), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Losses')
    plt.show()

    # Plot test predictions vs ground truth
    plt.figure(figsize=(8, 4))
    plt.scatter(test_labels_all, test_preds_all, alpha=0.5)
    plt.plot([test_labels_all.min(), test_labels_all.max()],
             [test_labels_all.min(), test_labels_all.max()],
             'k--', lw=2)
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.title('Test Predictions vs Ground Truth')
    plt.show()

    # Plot train predictions vs ground truth
    plt.figure(figsize=(10, 5))
    plt.scatter(train_labels_all, train_preds_all, alpha=0.5)
    plt.plot([train_labels_all.min(), train_labels_all.max()],
             [train_labels_all.min(), train_labels_all.max()],
             'k--', lw=2)
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.title('Train Predictions vs Ground Truth')
    plt.show()

if __name__ == "__main__":
    main()
