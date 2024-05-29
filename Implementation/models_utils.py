import torch
import numpy as np

# Constants for managing the size and iterations in the receptive field calculations
MAX_POINTS = 1024  # Maximum number of points in a receptive field for analysis
MAX_ITERATIONS = 10  # Maximum number of iterations allowed for expanding the receptive fields

def load_point_cloud(file_path: str) -> torch.Tensor:
    """
    Loads a point cloud from a specified file and returns it as a PyTorch tensor.
    
    Args:
        file_path (str): The path to the file containing the point cloud data.
    
    Returns:
        torch.Tensor: A tensor representation of the point cloud.
    """
    return torch.load(file_path)

def calculate_pairwise_distances(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates pairwise Euclidean distances between all points in each batch of point clouds.

    Args:
        points (torch.Tensor): A tensor of shape (B, N, D) where B is batch size,
                               N is number of points, and D is dimensionality of points.
    
    Returns:
        tuple: A tuple containing tensors of vector differences and their Euclidean distances.
    """
    B, N, _ = points.shape
    points_1 = points.unsqueeze(2)
    points_2 = points.unsqueeze(1)
    diffs = points_1 - points_2
    distances = torch.norm(diffs, dim=-1)
    return diffs, distances

def find_closest_neighbors(distances: torch.Tensor, num_neighbors: int) -> torch.Tensor:
    """
    Identifies the closest neighbors for each point within each point cloud based on the computed distances.

    Args:
        distances (torch.Tensor): A tensor of distances of shape (B, N, N), where B is batch size and N is number of points.
        num_neighbors (int): The number of closest neighbors to identify for each point.

    Returns:
        torch.Tensor: A tensor containing the indices of the nearest neighbors for each point.
    """
    B, N, _ = distances.shape
    if num_neighbors >= N:
        raise ValueError(f"Number of neighbors ({num_neighbors}) is greater than or equal to the number of points ({N}).")
    _, nearest_indices = torch.topk(distances, k=num_neighbors + 1, largest=False)
    nearest_indices = nearest_indices[:, :, 1:]  # Exclude self from neighbors
    return nearest_indices

def compute_geometric_features(points: torch.Tensor, num_neighbors: int, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes geometric features for each point in a batch of point clouds based on its closest neighbors.

    Args:
        points (torch.Tensor): Tensor containing batches of point clouds, shaped (B, N, D).
        num_neighbors (int): Number of nearest neighbors to consider for feature computation.
        batch_size (int): Number of point clouds to process in one go to manage memory efficiency.

    Returns:
        tuple: Geometric features and indices of nearest neighbors.
    """
    B, N, _ = points.shape
    geom_features_list = []
    nearest_indices_list = []

    for i in range(0, B, batch_size):
        points_batch = points[i:i + batch_size]
        diffs, distances = calculate_pairwise_distances(points_batch)
        nearest_indices = find_closest_neighbors(distances, num_neighbors)

        batch_indices = torch.arange(points_batch.shape[0]).view(points_batch.shape[0], 1, 1).expand(points_batch.shape[0], N, num_neighbors).to(points.device)
        vector_diffs = diffs[batch_indices, nearest_indices]
        scalar_diffs = distances[batch_indices, nearest_indices]

        scalar_diffs = scalar_diffs.unsqueeze(-1)
        geom_features = torch.cat((vector_diffs, scalar_diffs), dim=-1)

        geom_features_list.append(geom_features)
        nearest_indices_list.append(nearest_indices)

        del points_batch, diffs, distances, vector_diffs, scalar_diffs
        torch.cuda.empty_cache()

    geom_features = torch.cat(geom_features_list, dim=0)
    nearest_indices = torch.cat(nearest_indices_list, dim=0)

    return geom_features, nearest_indices

def determine_receptive_fields(nearest_indices: np.ndarray, chosen_indices: np.ndarray) -> list:
    """
    Expands and determines the receptive fields for given points based on their nearest neighbors.

    Args:
        nearest_indices (np.ndarray): Array of indices representing the nearest neighbors for each point.
        chosen_indices (np.ndarray): Array of starting indices from which to expand receptive fields.

    Returns:
        list: A list of lists, where each sublist contains the indices of points within the receptive field of a chosen point.
    """
    receptive_fields = []
    for chosen_idx in chosen_indices:
        field = set([chosen_idx])
        previous_field = field
        iteration_count = 0
        while len(field) < MAX_POINTS and iteration_count < MAX_ITERATIONS * 2:
            new_field = set()
            for idx in previous_field:
                new_field.update(list(nearest_indices[idx]))
            field.update(new_field)
            previous_field = new_field
            iteration_count += 1
        field = list(field)[:MAX_POINTS] # Ensure the field does not exceed maximum size
        receptive_fields.append(field)

    return receptive_fields

def get_receptive_fields(points: torch.Tensor, nearest_indices: torch.Tensor, chosen_indices: torch.Tensor) -> list:
    """
    Retrieves the receptive fields for selected points within point clouds, leveraging the nearest indices.

    Args:
        points (torch.Tensor): Tensor containing point clouds data.
        nearest_indices (torch.Tensor): Tensor of nearest neighbor indices for each point in the point clouds.
        chosen_indices (torch.Tensor): Tensor containing indices of points to calculate receptive fields for.

    Returns:
        list: List of tensors, each tensor representing the receptive field for a chosen point.
    """
    B, N, _ = points.shape
    chosen_indices = chosen_indices.unsqueeze(0).expand(B, -1) # Expand indices for batch processing
    receptive_fields = determine_receptive_fields(nearest_indices.cpu().numpy(), chosen_indices.cpu().numpy())
    receptive_fields = [points[torch.tensor(rf, dtype=int).to(points.device)] for rf in receptive_fields]
    return receptive_fields

def calculate_scaling_factors(train_data, feature_extractor, batch_size=1):
    """
    Calculates the mean and standard deviation of features extracted from training data for normalization.

    Args:
        train_data (iterable): Batch of training data.
        feature_extractor (model): Model used to extract features from the data.
        batch_size (int): Number of samples to process in one batch.

    Returns:
        tuple: Mean (mu) and standard deviation (sigma) of extracted features across all training data.
    """
    all_features = []

    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i + batch_size]
        features = feature_extractor.extract_features(batch)
        all_features.append(features.cpu().detach().numpy()) # Detach and move to CPU to prevent memory overflow

        del batch, features # Ensure memory is freed after processing
        torch.cuda.empty_cache() # Clear CUDA cache to free up unused memory

    all_features = np.concatenate(all_features, axis=0) 
    mu = np.mean(all_features, axis=0)
    sigma = np.std(all_features, axis=0)

    return mu, sigma

def detect_anomalies(student_model, teacher_model, test_data, mu, sigma, batch_size=1):
    """
    Detects anomalies in test data by comparing features extracted by student and teacher models.

    Args:
        student_model (model): Trained student model for feature extraction.
        teacher_model (model): Trained teacher model for reference feature extraction.
        test_data (iterable): Data on which anomalies are to be detected.
        mu (np.array): Mean of features from training data.
        sigma (np.array): Standard deviation of features from training data.
        batch_size (int): Number of samples to process in one batch.

    Returns:
        np.array: Array of anomaly scores, higher scores indicating higher likelihood of anomaly.
    """
    teacher_model.eval()
    student_model.eval()
    anomaly_scores = []

    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i + batch_size].to(student_model.device)

        with torch.no_grad():
            teacher_features, _ = teacher_model(batch)
            student_features, _ = student_model(batch)

        sigma_inv = torch.diag(torch.from_numpy(1.0 / sigma).to(teacher_features.device)) # Prepare inverse sigma for normalization
        teacher_features_normalized = (teacher_features - torch.from_numpy(mu).to(teacher_features.device)) @ sigma_inv
        regression_errors = torch.norm(student_features - teacher_features_normalized, dim=-1)
        anomaly_scores.append(regression_errors.cpu().numpy())

        del batch, teacher_features, student_features, regression_errors # Free memory after processing each batch
        torch.cuda.empty_cache()

    return np.concatenate(anomaly_scores)
def chamfer_distance(pc1, pc2):
    """
    Compute the Chamfer Distance between two point clouds pc1 and pc2.
    Args:
        pc1: Point cloud 1, a tensor of shape (N, 3)
        pc2: Point cloud 2, a tensor of shape (M, 3)
    Returns:
        Chamfer distance between pc1 and pc2.
    """
    # Expand each point cloud into a matrix form where every point from one cloud is paired with every point from the other
    diff = pc1.unsqueeze(1) - pc2.unsqueeze(0) # Broadcasting to create pairwise difference between points of pc1 and pc2
    dist = torch.norm(diff, dim=-1) # Compute the Euclidean distance for each pair

    # Compute the minimum distance for each point in pc1 to points in pc2 and vice versa
    min_dist_pc1 = torch.min(dist, dim=1)[0] # Minimum distance from each point in pc1 to any point in pc2
    min_dist_pc2 = torch.min(dist, dim=0)[0] # Minimum distance from each point in pc2 to any point in pc1

    # Calculate the Chamfer distance as the mean of all the minimum distances from both point clouds
    return torch.mean(min_dist_pc1) + torch.mean(min_dist_pc2) # Return the summed mean of minimum distances