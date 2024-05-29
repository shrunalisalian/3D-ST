import random
import torch
from pytorch3d.transforms import Rotate, random_rotations

def scale_to_unit_bounding_box(verts):
    """
    Scales the vertices of a 3D model so that it fits within a unit bounding box.

    Args:
        verts (torch.Tensor): Tensor containing the vertices of the model.

    Returns:
        torch.Tensor: Scaled vertices.
    """
    max_side = torch.max(verts.max(0)[0] - verts.min(0)[0]) # Find the longest side of the bounding box
    scale_factor = 1.0/ max_side # Calculate the scale factor to normalize the model
    verts = verts * scale_factor # Scale the vertices to fit them in a unit cube
    return verts

def rotate_object_randomly(verts):
    """
    Applies a random rotation to the vertices of a 3D model.

    Args:
        verts (torch.Tensor): Tensor containing the vertices of the model.

    Returns:
        torch.Tensor: Rotated vertices.
    """
    R = random_rotations(1).squeeze(0) # Generate a random rotation matrix
    verts = verts @ R.T # Apply the rotation to the vertices
    return verts

def place_object_randomly(verts):
    """
    Translates the vertices of a 3D model to a random location within a predefined range.

    Args:
        verts (torch.Tensor): Tensor containing the vertices of the model.

    Returns:
        torch.Tensor: Translated vertices.
    """
    translation = torch.FloatTensor(1,3).uniform_(-3,3) # Generate a random translation vector within [-3, 3]
    verts = verts + translation # Apply the translation to the vertices
    return verts

def generate_synthetic_scene(models):
    """
    Generates a synthetic scene by randomly placing, scaling, and rotating multiple models.

    Args:
        models (list): List of tuples, where each tuple contains vertices and faces of a model.

    Returns:
        Tuple of (torch.Tensor, torch.Tensor): Vertices and faces of the synthetic scene.
    """
    scene_verts = []
    scene_faces = []
    for _ in range(10): # Generate 10 models in the scene
        verts, faces = random.choice(models) # Randomly select a model from the list
        verts = scale_to_unit_bounding_box(verts) # Scale the model
        verts = rotate_object_randomly(verts) # Rotate the model
        verts = place_object_randomly(verts) # Translate the model
        scene_verts.append(verts) # Collect vertices for all models in the scene
        scene_faces.append(faces) # Collect faces for all models in the scene
    scene_verts = torch.cat(scene_verts, dim = 0) # Concatenate all vertices into a single tensor
    scene_faces = torch.cat(scene_faces, dim = 0) # Concatenate all faces into a single tensor
    return scene_verts, scene_faces


if __name__ == "__main__":
    from load_modelnet10 import load_modelnet10

    modelnet10_path = '/home/salian.sh/3D_Point_Cloud_Anomaly_Detection/ModelNet10'
    models = load_modelnet10(modelnet10_path) # Load models from ModelNet10

    scene_verts, scene_faces = generate_synthetic_scene(models) # Generate the synthetic scene
    print(f"Generate scene with {scene_verts.shape[0]} vertices")
    