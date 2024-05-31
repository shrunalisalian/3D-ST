# Implementation Folder README

This folder contains several Python scripts that are part of a project involving machine learning models for anomaly detection in point clouds. Below is a description of each file's purpose and functionality.

## Directory Structure

- `evaluation.py`: Contains code for evaluating the trained models against test datasets. It includes functions for calculating metrics such as Precision, Recall, and F1-Score.
- `models_utils.py`: Includes utility functions used across the project. This file contains essential functions like computing geometric features, scaling factors, and anomaly scores.
- `models.py`: Defines the neural network models used in the project. This includes the architecture of feature extractors and point cloud decoders.
- `train_student.py`: Contains the training loop for the student model. It uses data loaded from predefined datasets and trains the student model using backpropagation.
- `train_teacher.py`: Similar to `train_student.py`, but focuses on training the teacher model. It also sets up the teacher model's training parameters and supervises its training process.

## General Information

- **Python Version**: Python 3.8 or later is recommended for running these scripts.
- **External Libraries Required**:
  - `torch`: For model definition and operations.
  - `numpy`: For numerical operations.
  - `tqdm`: For displaying progress during operations.
  - `scipy`: Specifically for interpolation methods used in `evaluation.py`.

## Usage

Each script is executable as a standalone Python program. Ensure you have the required dependencies installed:

```bash
pip install torch numpy tqdm scipy
