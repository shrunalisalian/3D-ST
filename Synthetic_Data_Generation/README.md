# Synthetic Data Generation Folder README

This directory contains scripts and data files used for generating and handling synthetic datasets based on the ModelNet10 dataset. These datasets are primarily used for training and validating machine learning models that operate on 3D point clouds.

## Directory Contents

- `create_point_clouds.py`: This script contains functions for creating point clouds from the vertices of 3D models. It typically involves sampling methods such as farthest point sampling to reduce the number of points while maintaining the shape's characteristics.

- `generate_synthetic_scene.py`: This script is responsible for generating synthetic scenes by placing multiple 3D models within a single scene. Models are randomly scaled, rotated, and translated to create varied configurations.

- `load_modelnet10.py`: A utility script used to load 3D models from the ModelNet10 dataset. It processes the dataset to make it suitable for further operations like scene generation and point cloud creation.

- `training_set.pt`: A PyTorch tensor file containing the training dataset. This dataset is used to train machine learning models on tasks related to 3D point clouds.

- `validation_set.pt`: Similar to `training_set.pt`, this PyTorch tensor file contains the validation dataset used to evaluate the performance of the trained models and ensure that they generalize well to unseen data.

## Usage

To generate a new dataset or modify how the data is processed, you may run the Python scripts directly. Ensure you have the necessary Python environment and dependencies installed, including PyTorch and PyTorch3D.

Example command to run a script:
```bash
python create_point_clouds.py
