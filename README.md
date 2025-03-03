3D Anomaly Detection Implementation

Description

This repository contains my implementation of the 3D Student-Teacher (3D-ST) method for anomaly detection in 3D point clouds, as outlined in the assigned research paper for the Computer Vision Engineer position at Pivot Robots. The task involved creating synthetic datasets, implementing neural network models, and training these models using self-supervised and knowledge distillation techniques to detect anomalies in 3D point cloud data.

Repository Structure

Synthetic_Data_Generation/: Scripts to generate synthetic 3D scenes using the ModelNet10 dataset. Includes scripts for loading the dataset, generating synthetic scenes, and creating point clouds.
Implementation/: Python scripts for the implementation of the Teacher and Student models, including the training and evaluation stages.
models.py: Definitions of the Teacher and Student neural network architectures.
train_teacher.py: Script for training the Teacher model using synthetic data.
train_student.py: Script for training the Student model via knowledge distillation.
evaluation.py: Script to evaluate the models and visualize anomalies in 3D point clouds.
models_utils.py: Utility functions supporting model operations, including metrics and data transformations.

Key Features

Synthetic Data Generation: Automated generation of synthetic datasets from 3D models, ensuring robust training capable of covering various scenarios.
Deep Learning Models: Implementation of complex neural network architectures as specified in the paper, focusing on the adaptation of the student-teacher framework for 3D data.
Self-Supervised Learning: Leveraging a novel approach to pretrain the Teacher model, enhancing its capability to extract meaningful geometric descriptors from 3D point clouds.
Anomaly Detection: Utilizing regression errors between the Teacher and Student models' outputs to reliably localize anomalous structures in test data.


Technologies Used

Python 3.8+
PyTorch 1.7+ (including PyTorch3D for 3D operations)
NumPy, SciPy for numerical operations
Matplotlib, Plotly for visualization

Setup and Run
Clone the repository: git clone https://github.com/shrunalisalian/3D-ST

Install dependencies: pip install -r requirements.txt
Generate the synthetic data: python Synthetic_Data_Generation/create_point_clouds.py

Train the models:
python Implementation/train_teacher.py
python Implementation/train_student.py

Evaluate the models: python Implementation/evaluation.py

