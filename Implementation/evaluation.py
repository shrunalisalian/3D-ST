import os
import torch
import numpy as np
from tqdm import tqdm
from models_ROUGH import FeatureExtractor, StudentNetwork
from models_utils import compute_geometric_features, calculate_scaling_factors, chamfer_distance
from scipy.interpolate import griddata

# Define constants for device selection and model configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU
NUM_NEIGHBORS = 20  # Number of nearest neighbors to use in feature extraction
FEATURE_DIM = 128  # Dimension of the feature vectors in the model
NUM_RESIDUAL_BLOCKS = 3  # Number of residual blocks in the network architecture
N_POINTS = 64000  # Number of input points


def load_model(model_class, model_path, device, **kwargs):
    """
    Loads a model from a specified file.

    Args:
        model_class (class): Class of the model to be loaded.
        model_path (str): Path to the model's state dictionary.
        device (torch.device): Device on which the model should be loaded.
        **kwargs: Additional arguments for model initialization.

    Returns:
        nn.Module: Loaded and initialized model ready for inference.
    """
    model = model_class(**kwargs).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval() # Set the model to evaluation mode
    return model

def compute_anomaly_scores(teacher_model, student_model, test_data, mu, sigma, batch_size=1):
    """
    Computes anomaly scores for a dataset using the teacher and student models.

    Args:
        teacher_model, student_model (torch.nn.Module): Pretrained models for feature extraction.
        test_data (torch.Tensor): Data to be tested.
        mu, sigma (np.array): Mean and standard deviation for normalization.
        batch_size (int): Number of samples to process at once.

    Returns:
        np.array: Array of anomaly scores.
    """
    anomaly_scores = []
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i + batch_size].to(DEVICE)

        with torch.no_grad():
            teacher_features = teacher_model.extract_features(batch)
            student_features, _ = student_model(batch)

        sigma_inv = torch.diag(torch.from_numpy(1.0 / sigma).to(DEVICE))
        teacher_features_normalized = (teacher_features - torch.from_numpy(mu).to(DEVICE)) @ sigma_inv
        regression_errors = torch.norm(student_features - teacher_features_normalized, dim=-1)
        anomaly_scores.append(regression_errors.cpu().numpy())

        del batch, teacher_features, student_features, regression_errors
        torch.cuda.empty_cache()

    return np.concatenate(anomaly_scores)

def harmonic_interpolation(points, values, grid_size=64000):
    """
    Applies harmonic interpolation to anomaly scores over a grid defined by the point cloud extents.

    Args:
        points (np.array): Coordinates of points in the point cloud.
        values (np.array): Anomaly scores associated with each point.
        grid_size (int): Resolution of the grid for interpolation.

    Returns:
        np.array: Interpolated values over the grid.
    """
    grid_x, grid_y, grid_z = np.mgrid[
        np.min(points[:, 0]):np.max(points[:, 0]):grid_size * 1j,
        np.min(points[:, 1]):np.max(points[:, 1]):grid_size * 1j,
        np.min(points[:, 2]):np.max(points[:, 2]):grid_size * 1j
    ]
    grid_values = griddata(points, values, (grid_x, grid_y, grid_z), method='harmonic')
    return grid_values

def evaluate_pro(anomaly_scores, ground_truth, thresholds):
    """
    Evaluates the performance of anomaly detection using the Per-Region Overlap (PRO) metric.

    Args:
        anomaly_scores (np.array): Anomaly scores where higher values indicate anomalies.
        ground_truth (np.array): Binary array indicating true anomaly locations.
        thresholds (np.array): Set of thresholds to evaluate against.

    Returns:
        list: PRO values for each threshold.
    """
    pro_values = []
    for threshold in thresholds:
        binary_anomalies = anomaly_scores > threshold
        tp = np.sum(binary_anomalies & ground_truth)
        fp = np.sum(binary_anomalies & ~ground_truth)
        fn = np.sum(~binary_anomalies & ground_truth)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        pro = (precision + recall) / 2
        pro_values.append(pro)
    return pro_values

def main():
    # Load the test dataset
    test_data_path = 'path_to_test_data/test_data.pt'
    test_data = torch.load(test_data_path).float().to(DEVICE)

    # Print original shape to verify the data structure
    print(f"Test data shape: {test_data.shape}")

    # Load the pretrained teacher model
    teacher_model_path = 'path_to_teacher_model/teacher_model_weights.pth'
    teacher_model = load_model(FeatureExtractor, teacher_model_path, DEVICE, neighbors=NUM_NEIGHBORS, dim=FEATURE_DIM, num_res_blocks=NUM_RESIDUAL_BLOCKS)

    # Load the pretrained student model
    student_model_path = 'path_to_student_model/student_model_weights.pth'
    student_model = load_model(StudentNetwork, student_model_path, DEVICE, neighbors=NUM_NEIGHBORS, dim=FEATURE_DIM, num_res_blocks=NUM_RESIDUAL_BLOCKS)

    # Calculate scaling factors
    mu, sigma = calculate_scaling_factors(test_data, teacher_model)

    # Compute anomaly scores
    anomaly_scores = compute_anomaly_scores(teacher_model, student_model, test_data, mu, sigma)
    print("Computed anomaly scores.")

    # Apply harmonic interpolation
    points = test_data.cpu().numpy().reshape(-1, 3)
    interpolated_anomaly_scores = harmonic_interpolation(points, anomaly_scores)
    print("Applied harmonic interpolation.")

    # Load ground truth for evaluation (assuming ground truth is available)
    ground_truth_path = 'path_to_ground_truth/ground_truth.pt'
    ground_truth = torch.load(ground_truth_path).cpu().numpy()

    # Evaluate using PRO
    thresholds = np.linspace(0, 1, 100)
    pro_values = evaluate_pro(interpolated_anomaly_scores, ground_truth, thresholds)
    au_pro = np.trapz(pro_values, dx=thresholds[1] - thresholds[0])
    print(f"Area under PRO curve: {au_pro}")

if __name__ == "__main__":
    main()
