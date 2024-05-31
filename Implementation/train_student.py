import os
import torch
import numpy as np
from tqdm import tqdm
from models_ROUGH import FeatureExtractor, StudentNetwork
from models_utils import compute_geometric_features, calculate_scaling_factors

# Define constants for device selection and training parameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Automatically use GPU if available
NUM_TRAINING_EPOCHS = 100  # Total number of training epochs
TRAINING_LR = 1e-4  # Learning rate for training
TRAINING_WD = 1e-4  # Weight decay for regularization during training
HIDDEN_LAYER_DIM = 128  # Number of dimensions in hidden layers of the network
LEAKY_RELU_SLOPE = 0.2  # Slope coefficient for LeakyReLU activations
NUM_OUTPUT_POINTS = 2048  # Number of output points from the decoder

def calculate_fmap_distribution(features):
    """
    Calculates the standard deviation and mean of flattened feature maps.

    Args:
        features (torch.Tensor): Feature maps from the neural network.

    Returns:
        Tuple containing standard deviation and mean of the feature maps.
    """
    flattened_features = features.view(-1, features.size(-1)).to(DEVICE) # Flatten and move to the correct device
    std, mean = torch.std_mean(flattened_features, dim=0) # Calculate standard deviation and mean
    return std.cpu(), mean.cpu() # Return values to CPU for further processing

def train_student(train_data, val_data, student, teacher, optimizer, mu, sigma, num_epochs):
    """
    Trains the student model using data, teacher outputs, and calculated scaling factors.

    Args:
        train_data, val_data (torch.Tensor): Datasets for training and validation.
        student (nn.Module): Student model to be trained.
        teacher (nn.Module): Teacher model for comparison.
        optimizer (torch.optim.Optimizer): Optimizer for training the student model.
        mu, sigma (torch.Tensor): Mean and standard deviation for feature normalization.
        num_epochs (int): Total number of epochs for training.

    Returns:
        Lists of training and validation losses for each epoch.
    """
    student.train() # Set student model to training mode
    training_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        epoch_training_loss = 0.0
        epoch_validation_loss = 0.0

        # Training loop with progress display
        for point_cloud in tqdm(train_data, desc=f'Training Epoch {epoch+1}/{num_epochs}'):
            optimizer.zero_grad() # Clear gradients
            point_cloud = point_cloud.to(DEVICE) # Move data to the correct device
            geom_features, neighbor_idxs = compute_geometric_features(point_cloud, student.feature_extractor.neighbors)
            student_features = student(point_cloud, geom_features, neighbor_idxs) # Get student's output

            with torch.no_grad(): # No gradient calculation for teacher's pass
                teacher_features = teacher(point_cloud, geom_features, neighbor_idxs) # Get teacher's output

            sigma_inv = torch.diag(1.0 / sigma).to(DEVICE) # Prepare normalized teacher features
            teacher_features_normalized = (teacher_features - mu.to(DEVICE)) @ sigma_inv
            loss = student.calculate_loss(student_features, teacher_features_normalized, mu, sigma)
            loss.backward() # Backpropagation
            optimizer.step() # Update weights

            epoch_training_loss += loss.item() # Sum up loss for the epoch

        # Validation loop with progress display
        with torch.no_grad(): # Disable gradient computation in validation phase
            for point_cloud in tqdm(val_data, desc=f'Validation Epoch {epoch+1}/{num_epochs}'):
                point_cloud = point_cloud.to(DEVICE)
                geom_features, neighbor_idxs = compute_geometric_features(point_cloud, student.feature_extractor.neighbors)
                student_features = student(point_cloud, geom_features, neighbor_idxs)
                teacher_features = teacher(point_cloud, geom_features, neighbor_idxs)

                sigma_inv = torch.diag(1.0 / sigma).to(DEVICE)
                teacher_features_normalized = (teacher_features - mu.to(DEVICE)) @ sigma_inv
                loss = student.calculate_loss(student_features, teacher_features_normalized, mu, sigma)

                epoch_validation_loss += loss.item() # Sum up validation loss

        # Record average losses for this epoch
        training_losses.append(epoch_training_loss / len(train_data))
        validation_losses.append(epoch_validation_loss / len(val_data))

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {training_losses[-1]:.4f}, Validation Loss: {validation_losses[-1]:.4f}')

    return training_losses, validation_losses

def main():
    """
    Main function to load data, create models, train the student model, and save the trained model.
    """
    # Load training and validation datasets
    train_data = torch.load('../Synthetic_Data_Generation/training_set.pt')
    val_data = torch.load('../Synthetic_Data_Generation/validation_set.pt')

    # Move data to the correct device
    train_data = train_data.float().to(DEVICE)
    val_data = val_data.float().to(DEVICE)

    # Load the teacher model, configure it for evaluation
    teacher = FeatureExtractor(neighbors=20, dim=128, num_res_blocks=3).to(DEVICE)
    teacher.load_state_dict(torch.load('teacher_model.pth'))
    teacher.eval()

    # Calculate mean and standard deviation of feature maps from the training data
    mu, sigma = calculate_scaling_factors(train_data, teacher)

    # Create the student model and optimizer
    student = StudentNetwork(neighbors=20, dim=128, num_res_blocks=3).to(DEVICE)
    optimizer = torch.optim.Adam(student.parameters(), lr=TRAINING_LR, weight_decay=TRAINING_WD)

    # Train the student model and save it
    train_student(train_data, val_data, student, teacher, optimizer, mu, sigma, NUM_TRAINING_EPOCHS)
    torch.save(student.state_dict(), 'student_model.pth') # Save the trained student model

if __name__ == "__main__":
    main()
