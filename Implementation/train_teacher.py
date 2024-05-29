import os
import torch
import numpy as np
from tqdm import tqdm
from models_ROUGH import FeatureExtractor, PointCloudDecoder
from models_utils import compute_geometric_features, chamfer_distance, calculate_scaling_factors

# Define constants for device selection, pretraining parameters, and decoder settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, else CPU
NUM_PRETRAINING_EPOCHS = 100  # Number of epochs for pretraining
PRETRAINING_LR = 1e-4  # Learning rate for pretraining
PRETRAINING_WD = 1e-4  # Weight decay for regularization during pretraining
DECODER_SAMPLE_SIZE = 1024  # Number of samples to decode from the point cloud
HIDDEN_LAYER_DIM = 128  # Dimensionality of hidden layers in MLPs
LEAKY_RELU_SLOPE = 0.2  # Slope for LeakyReLU activation functions
NUM_OUTPUT_POINTS = 2048  # Number of points to output by the decoder

# Model pass function
def model_pass(teacher, decoder, point_cloud):
    """
    Conducts a single forward and backward pass for one point cloud, including feature extraction, decoding, and loss calculation.

    Args:
        teacher (FeatureExtractor): The feature extraction model.
        decoder (PointCloudDecoder): The point cloud decoding model.
        point_cloud (torch.Tensor): A single point cloud tensor.

    Returns:
        Tuple containing the loss tensor and the extracted point features.
    """
    point_cloud = point_cloud.to(DEVICE)  # Move point cloud to the correct device
    geom_features, neighbor_idxs = compute_geometric_features(point_cloud, teacher.neighbors)
    point_features = teacher(point_cloud, geom_features, neighbor_idxs)  # Extract features

    chosen_idxs = torch.randint(0, point_cloud.size(1), (DECODER_SAMPLE_SIZE,))  # Randomly select indices for decoding
    chosen_point_features = point_features[chosen_idxs]  # Get features for selected indices
    decoded_points = decoder(chosen_point_features)  # Decode selected features back to point cloud

    loss = chamfer_distance(point_cloud[chosen_idxs], decoded_points)  # Calculate Chamfer distance as loss
    return loss, point_features

# Training loop
def train(train_data, val_data, teacher, decoder, optimizer, num_epochs, save_dir):
    """
    Runs the training loop over specified epochs, saving model states and losses.

    Args:
        train_data, val_data (torch.Tensor): Training and validation datasets.
        teacher (FeatureExtractor), decoder (PointCloudDecoder): Models to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        num_epochs (int): Total number of epochs to train for.
        save_dir (str): Directory to save training outputs and model states.
    """
    pbar = tqdm(range(num_epochs), position=0, leave=True) # Progress bar for tracking training progress
    training_losses = []
    validation_losses = []

    for epoch in pbar:
        epoch_training_loss = 0.0
        epoch_validation_loss = 0.0

        teacher.train() # Set models to training mode
        decoder.train()
        for point_cloud in train_data:
            # Debugging: Print the shape of point_cloud
            print(f"Training epoch {epoch}, shape of point_cloud: {point_cloud.shape}")

            optimizer.zero_grad()
            loss, _ = model_pass(teacher, decoder, point_cloud)
            loss.backward() # Backpropagate the loss
            optimizer.step() # Update model parameters
            epoch_training_loss += loss.item() # Accumulate the loss

        teacher.eval() # Set models to evaluation mode
        decoder.eval()
        with torch.no_grad(): # Disable gradient computation for validation
            for point_cloud in val_data:
                # Debugging: Print the shape of point_cloud
                print(f"Validation epoch {epoch}, shape of point_cloud: {point_cloud.shape}")

                loss, _ = model_pass(teacher, decoder, point_cloud)
                epoch_validation_loss += loss.item() # Accumulate the validation loss

        # Record and print losses for each epoch
        training_losses.append(epoch_training_loss / len(train_data))
        validation_losses.append(epoch_validation_loss / len(val_data))
        pbar.set_postfix({'train_loss': training_losses[-1], 'val_loss': validation_losses[-1]})

    # Save training and validation losses to disk
    np.save(os.path.join(save_dir, 'pre_training_losses.npy'), np.array(training_losses))
    np.save(os.path.join(save_dir, 'pre_validation_losses.npy'), np.array(validation_losses))
    torch.save(teacher.state_dict(), os.path.join(save_dir, "pretrained_teacher.pth"))
    torch.save(decoder.state_dict(), os.path.join(save_dir, "pretrained_decoder.pth"))

def main():
    """
    Main function to load data, create models, optimizer and run the training loop.
    """
    # Load pre-generated training and validation data sets
    train_data = torch.load('../Synthetic_Data_Generation/training_set.pt')
    val_data = torch.load('../Synthetic_Data_Generation/validation_set.pt')

    # Debugging: Print the shape of loaded data
    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")

    # Configure data tensors and move them to the specified device
    train_data = train_data.float().to(DEVICE)
    val_data = val_data.float().to(DEVICE)

    # Reshape data to conform to (B, N, 3), B: batch size, N: number of points, 3: coordinates
    train_data = train_data.view(-1, 2048, 3)
    val_data = val_data.view(-1, 2048, 3)

    # Create models and optimizer
    teacher = FeatureExtractor(neighbors=20, dim=128, num_res_blocks=3).to(DEVICE)
    decoder = PointCloudDecoder(dim=128, hidden_dim=HIDDEN_LAYER_DIM, leaky_slope=LEAKY_RELU_SLOPE, num_output_points=NUM_OUTPUT_POINTS).to(DEVICE)
    optimizer = torch.optim.Adam(list(teacher.parameters()) + list(decoder.parameters()), lr=PRETRAINING_LR, weight_decay=PRETRAINING_WD)

    # Start the training loop
    train(train_data, val_data, teacher, decoder, optimizer, NUM_PRETRAINING_EPOCHS, save_dir='../training_saves')

if __name__ == "__main__":
    main()

