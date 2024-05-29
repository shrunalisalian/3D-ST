# Import necessary libraries
import os
import torch 
import pytorch3d.io as io

def load_off(file_path):
    """
    Loads a 3D model from an OFF file format.

    Args:
        file_path (str): Path to the OFF file.

    Returns:
        Tuple containing:
        - verts (torch.Tensor): Vertices of the 3D model as a tensor of shape (n_verts, 3).
        - faces (torch.Tensor): Faces of the 3D model as a tensor of shape (n_faces, vertices per face).

    Raises:
        ValueError: If the file is not a valid OFF file.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if lines[0].strip() != 'OFF':
            raise ValueError('Not a valid OFF file')
        
        n_verts, n_faces, _ = map(int, lines[1].strip().split())
        verts = []
        for i in range(2, 2+ n_verts):
            verts.append(list(map(float, lines[i].strip().split())))

        faces = []
        for i in range(2 + n_verts, 2 + n_verts + n_faces):
            face_data = list(map(int, lines[i].strip().split()))
            faces.append(face_data[1:])

        verts = torch.tensor(verts, dtype = torch.float32)
        faces = torch.tensor(faces, dtype = torch.int64)

    return verts, faces

def load_modelnet10(modelnet10_path):
    """
    Loads all 3D models from the ModelNet10 dataset stored in a directory.

    Args:
        modelnet10_path (str): Root directory path where ModelNet10 is stored.

    Returns:
        List of tuples: Each tuple contains vertices and faces tensors of each model.
    """
    models = []
    categories = os.listdir(modelnet10_path)
    for category in categories:
        category_path = os.path.join(modelnet10_path, category)
        for split in ['train', 'test']:
            split_path = os.path.join(category_path, split)
            if os.path.exists(split_path):
                for file in os.listdir(split_path):
                    if file.endswith('.off'):
                        model_path = os.path.join(split_path, file)
                        verts, faces = load_off(model_path)
                        models.append((verts, faces))

    return models

if __name__ == '__main__':
    modelnet10_path = '/home/salian.sh/3D_Point_Cloud_Anomaly_Detection/ModelNet10'
    models = load_modelnet10(modelnet10_path)
    print(f"Loaded {len(models)} models from ModelNet10")

