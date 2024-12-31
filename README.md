# GNN + CNN Hybrid for Permeability Prediction

This repository contains a hybrid neural network architecture that combines:
- A **Graph Neural Network (GNN)** to handle graph-structured pore network data  
- A **Convolutional Neural Network (CNN)** to process 2D images  

The goal is to predict permeability (or similar properties) by leveraging both graph and image data.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Data Setup](#data-setup)
- [Running the Training Script](#running-the-training-script)
- [Notes](#notes)
- [License](#license)

---

## Project Structure

A simplified overview of the repository:


- **gnn_cnn/**: Package containing code for datasets, models, and the main training script.
- **data/data_inverse/**: Contains GNN data (graphs and labels).
- **DeePore_Compact_Data.h5**: Example HDF5 file containing image data used by the CNN.

You may also have additional scripts for pure GNN or pure CNN experiments.

---

## Requirements

You’ll need the following (or similar) packages:

- Python 3.8+  
- [PyTorch](https://pytorch.org/)  
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)  
- NumPy  
- Matplotlib  
- DeePore (either installed or placed as a local module)

For example, using conda:

```bash
conda create -n PyG python=3.8
conda activate PyG
pip install torch torchvision torchaudio
pip install torch-geometric
pip install numpy matplotlib

Data Setup
Graph Data (for GNN)

Place your graph files (graph_0.pt, graph_1.pt, etc.) in data/data_inverse/graphs/.
The file test_GT.pkl should be in data/data_inverse/ for labels.
Image Data (for CNN)

Ensure DeePore_Compact_Data.h5 (or your HDF5 file) is in your project root or a known location.
If it’s in a different folder, update the path in the script accordingly.
Folder Structure

If the script references data/data_inverse/graphs or data/data_inverse/test_GT.pkl, make sure your actual folders match these paths.
If you run into a FileNotFoundError, double-check the relative paths and your working directory.

Running the Training Script
Activate your environment (if using conda/virtualenv):

bash
Copy code
conda activate PyG
Navigate to the project root:

bash
Copy code
cd /path/to/GNN_CNN_PNM
Run the Hybrid Training:

bash
Copy code
python -m gnn_cnn.hybrid_train
This command invokes Python’s module mode, treating gnn_cnn as a package and running the hybrid_train.py script.
Make sure you have an __init__.py inside gnn_cnn/, so Python recognizes it as a package.
During training, the script will:

Load both the GNN dataset (graphs + labels) and the CNN dataset (images + labels).
Split them into train/test sets using matching indices.
Create a hybrid model (HybridModel) to process the graph and image data in parallel.
Print out training and testing losses each epoch.
Plot loss curves and predictions vs. ground truth at the end of training.
Notes
Path Adjustments: If you’ve moved or renamed data_inverse/ or your HDF5 file, be sure to update the paths in the script.
Hardware: If you have a CUDA-capable GPU, the script will use GPU (cuda) by default when torch.cuda.is_available() is True. Otherwise, it uses CPU.
Hyperparameters: The batch size, learning rate, number of epochs, and hidden layer dimensions can be tuned in hybrid_train.py.
Modularization: You can split out data loading, modeling, and training into separate files (datasets.py, models.py, train.py) for clarity.