import torch
from pytorch3d.ops import sample_farthest_points

def farthest_point_sampling(points, n_samples):
    """
    Samples the farthest points from a given point cloud using the farthest point sampling algorithm.

    Args:
        points (torch.Tensor): Tensor containing the vertices of the point cloud of shape (N, 3), where N is the number of points.
        n_samples (int): Number of points to sample.

    Returns:
        torch.Tensor: Sampled points of shape (n_samples, 3).
    """
    # Expand the dimensions to match the expected input [batch size, number of points, dimensions]
    sampled_verts, _ = sample_farthest_points(points.unsqueeze(0), K=n_samples)
    # Squeeze to remove batch dimension since batch size is 1
    return sampled_verts.squeeze(0)

def create_point_cloud(verts, n_points):
    """
    Creates a point cloud by sampling a specified number of points from the vertices of a model.

    Args:
        verts (torch.Tensor): Tensor containing vertices of the model's mesh.
        n_points (int): Number of points to sample.

    Returns:
        torch.Tensor: Tensor representing the sampled point cloud.
    """
    # Sample points using the farthest point sampling technique
    sampled_verts = farthest_point_sampling(verts, n_points)
    return sampled_verts

def generate_dataset(models, num_samples, n_points):
    """
    Generates a dataset by creating multiple synthetic scenes and sampling point clouds from them.

    Args:
        models (list): List of models, each containing vertices and possibly faces.
        num_samples (int): Number of samples to generate in the dataset.
        n_points (int): Number of points in each sampled point cloud.

    Returns:
        torch.Tensor: Tensor stack of all point clouds in the dataset.
    """
    from generate_synthetic_scene import generate_synthetic_scene
    
    dataset = []
    for _ in range(num_samples):
        # Generate a synthetic scene using the provided models
        scene_verts, _ = generate_synthetic_scene(models)
        # Sample a point cloud from the synthetic scene
        point_cloud = create_point_cloud(scene_verts, n_points)
        # Collect all sampled point clouds
        dataset.append(point_cloud)
    # Stack all point clouds into a single tensor for easy handling
    return torch.stack(dataset)

if __name__ == "__main__":
    from load_modelnet10 import load_modelnet10

    # Path to the ModelNet10 dataset directory
    modelnet10_path = '/home/salian.sh/3D_Point_Cloud_Anomaly_Detection/ModelNet10'
    # Load models from ModelNet10 dataset
    models = load_modelnet10(modelnet10_path)
    
    # Number of points in each point cloud
    n_points = 2048
    # Generate the training set
    training_set = generate_dataset(models, 500, n_points)
    # Generate the validation set
    validation_set = generate_dataset(models, 25, n_points)
    
    # Save the generated datasets
    torch.save(training_set, 'training_set.pt')
    torch.save(validation_set, 'validation_set.pt')
    
    # Output message confirming dataset generation and saving
    print("Training and validation sets generated and saved.")
