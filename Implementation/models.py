import torch
import torch.nn as nn
import torch.nn.functional as F
from models_utils import * # Import utility functions and classes such as MLP and compute_geometric_features

# Define constants
HIDDEN_LAYER_DIM = 128  # Number of dimensions in hidden layers of MLPs
LEAKY_RELU_SLOPE = 0.2  # Negative slope coefficient for LeakyReLU activations
NUM_OUTPUT_POINTS = 2048  # Number of output points for the decoder

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 128]):
        """
        Initializes the MLP class.

        Args:
            input_dim (int): The dimension of the input features.
            output_dim (int): The dimension of the output features.
            hidden_dims (list of int): List of dimensions for the hidden layers.
        """
        super(MLP, self).__init__()
        layers = []
        current_dim = input_dim

        # Create hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))  # Linear transformation
            layers.append(nn.LayerNorm(hidden_dim))  # Layer Normalization
            layers.append(nn.LeakyReLU(0.2))  # LeakyReLU activation
            current_dim = hidden_dim  # Update current_dim to the output of this layer

        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))  # Final Linear transformation
        self.mlp = nn.Sequential(*layers)  # Sequentially stack the layers

    def forward(self, x):
        """
        Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        return self.mlp(x)


# class LFA(nn.Module):
#     """
#     Local Feature Aggregation (LFA) module for point clouds.
#     """
#     def __init__(self, dim, hidden_dims=[64, 128]):
#         """
#         Initializes the LFA class.

#         Args:
#             dim (int): The dimension of the output features.
#             hidden_dims (list of int): List of dimensions for the hidden layers in the MLPs.
#         """
#         super(LFA, self).__init__()
#         self.dim = dim
#         # Geometric features MLP
#         self.geom_mlp = MLP(4, dim, hidden_dims=hidden_dims)
#         # Final MLP after concatenation
#         self.feature_mlp = MLP(dim * 2, dim, hidden_dims=hidden_dims)

#     def forward(self, point_features, geom_features, neighbor_idxs):
#         """
#         Forward pass through the LFA module.

#         Args:
#             point_features (torch.Tensor): Tensor containing point features, shaped (B, N, C).
#             geom_features (torch.Tensor): Tensor containing geometric features, shaped (B, N, k_neighbors, 4).
#             neighbor_idxs (torch.Tensor): Tensor containing indices of nearest neighbors, shaped (B, N, k_neighbors).

#         Returns:
#             torch.Tensor: Output tensor of aggregated features, shaped (B, N, dim).
#         """
#         B, N, k_neighbors = neighbor_idxs.shape
#         C = point_features.shape[-1]

#         print(f"B: {B}, N: {N}, k_neighbors: {k_neighbors}")
#         print(f"geom_features shape: {geom_features.shape}")

#         # Ensure point_features has shape (B, N, C)
#         if point_features.dim() == 2:
#             point_features = point_features.unsqueeze(0).expand(B, N, C)

#         # Compute geometric features
#         geom_features_reshaped = geom_features.view(B * N * k_neighbors, 4)  # Shape (B * N * k_neighbors, 4)
#         geom_outputs_reshaped = self.geom_mlp(geom_features_reshaped)  # Shape (B * N * k_neighbors, dim)
#         geom_outputs = geom_outputs_reshaped.view(B, N, k_neighbors, -1)  # Shape (B, N, k_neighbors, dim)
#         print(f"geom_outputs shape after MLP: {geom_outputs.shape}")

#         # Expand point_features to include the neighbors dimension
#         point_features_expanded = point_features.unsqueeze(2).expand(B, N, k_neighbors, C)
#         print(f"point_features_expanded shape: {point_features_expanded.shape}")

#         # Expand neighbor_idxs to include the feature dimension
#         neighbor_idxs_expanded = neighbor_idxs.unsqueeze(-1).expand(B, N, k_neighbors, C)
#         print(f"neighbor_idxs_expanded shape: {neighbor_idxs_expanded.shape}")

#         # Extract relevant input features using torch.gather
#         relevant_input_features = torch.gather(point_features_expanded, 2, neighbor_idxs_expanded)
#         print(f"relevant_input_features shape: {relevant_input_features.shape}")

#         # Ensure dimensions match before concatenation
#         geom_outputs_expanded = geom_outputs.unsqueeze(-1).expand_as(relevant_input_features)
#         print(f"geom_outputs shape after unsqueeze and expand: {geom_outputs_expanded.shape}")

#         # Concatenate geometric features and input features
#         combined_features = torch.cat((geom_outputs_expanded, relevant_input_features), dim=-1)
#         print(f"combined_features shape: {combined_features.shape}")

#         # Apply final MLP and average pooling
#         final_features = self.feature_mlp(combined_features.view(-1, combined_features.shape[-1]))  # Flatten last two dimensions for MLP
#         final_features = final_features.view(B, N, k_neighbors, -1)  # Reshape back to (B, N, k_neighbors, dim)
#         final_features = torch.mean(final_features, dim=2)  # Average pooling across neighbors
#         print(f"final_features shape after average pooling: {final_features.shape}")

#         return final_features

class LFA(nn.Module):
    """
    Local Feature Aggregation (LFA) module for point clouds.
    """
    def __init__(self, dim):
        super(LFA, self).__init__()
        self.dim = dim
        self.geom_mlp = MLP(4, dim)  # Geometric features MLP
        self.feature_mlp = MLP(dim * 2, dim)  # Final MLP after concatenation

    def forward(self, point_features, geom_features, neighbor_idxs):
        B, N, k_neighbors = neighbor_idxs.shape
        C = point_features.shape[-1]

        # Reshape geom_features before passing to geom_mlp
        geom_features_reshaped = geom_features.view(B * N * k_neighbors, -1)  # Shape (B * N * k_neighbors, 4)
        geom_outputs_reshaped = self.geom_mlp(geom_features_reshaped)  # Shape (B * N * k_neighbors, dim)
        geom_outputs = geom_outputs_reshaped.view(B, N, k_neighbors, -1)  # Shape (B, N, k_neighbors, dim)
        print(f"geom_outputs shape after MLP: {geom_outputs.shape}")

        # Ensure point_features has the correct shape before expansion
        if point_features.dim() == 2:
            point_features = point_features.unsqueeze(0)  # Shape (1, N, C)
        assert point_features.shape == (B, N, C), f"point_features shape mismatch: {point_features.shape}, expected ({B}, {N}, {C})"

        # Expand point_features to include the neighbors dimension
        point_features_expanded = point_features.unsqueeze(2).expand(B, N, k_neighbors, C)  # Shape (B, N, k_neighbors, C)
        print(f"point_features_expanded shape: {point_features_expanded.shape}")

        # Gather relevant input features using neighbor_idxs
        relevant_input_features = torch.gather(point_features_expanded, 1, neighbor_idxs.unsqueeze(-1).expand(B, N, k_neighbors, C))  # Shape (B, N, k_neighbors, C)
        print(f"relevant_input_features shape: {relevant_input_features.shape}")

        # Concatenate geometric features and input features
        combined_features = torch.cat((geom_outputs, relevant_input_features), dim=-1)  # Shape (B, N, k_neighbors, dim + C)
        print(f"combined_features shape: {combined_features.shape}")

        # Apply final MLP and average pooling
        final_features = self.feature_mlp(combined_features)  # Shape (B, N, k_neighbors, dim)
        final_features = torch.mean(final_features, dim=2)  # Average pooling, Shape (B, N, dim)
        print(f"final_features shape after average pooling: {final_features.shape}")

        return final_features



class ResidualBlock(nn.Module):
    """
    A residual block that uses MLPs for transformation and Local Feature Aggregation (LFA).
    """
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.mlp_initial = MLP(dim, dim // 4)  # Initial transformation reducing dimensionality
        self.lfa1 = LFA(dim // 4)  # First local feature aggregation
        self.lfa2 = LFA(dim // 2)  # Second local feature aggregation
        self.mlp_final = MLP(dim, dim)  # Final MLP to transform back to original dimensionality
        self.mlp_residual = MLP(dim, dim)  # MLP for processing the residual connection

    def forward(self, input_features, geom_features, neighbor_idxs):
        print(f"Input features shape: {input_features.shape}")
        print(f"Geometric features shape: {geom_features.shape}")
        print(f"Neighbor indices shape: {neighbor_idxs.shape}")
        
        residual = self.mlp_residual(input_features)
        print(f"Residual features shape: {residual.shape}")
        
        activated = self.mlp_initial(input_features)
        print(f"After initial MLP, features shape: {activated.shape}")
        
        activated = self.lfa1(activated, geom_features, neighbor_idxs)
        print(f"After first LFA, features shape: {activated.shape}")
        
        activated = self.lfa2(activated, geom_features, neighbor_idxs)
        print(f"After second LFA, features shape: {activated.shape}")
        
        activated = self.mlp_final(activated)
        print(f"After final MLP, features shape: {activated.shape}")
        
        combined_features = activated + residual
        print(f"Combined features shape: {combined_features.shape}")
        
        return combined_features




class FeatureExtractor(nn.Module):
    """
    Feature extractor composed of multiple ResidualBlocks for extracting point cloud features.
    """
    def __init__(self, neighbors, dim, num_res_blocks):
        # Initialize the FeatureExtractor with multiple ResidualBlocks
        super(FeatureExtractor, self).__init__()
        self.neighbors = neighbors # Number of neighbors to consider for feature aggregation
        self.dim = dim # Dimensionality of the feature space
        self.res_blocks = nn.ModuleList([ResidualBlock(dim) for _ in range(num_res_blocks)]) # List of residual blocks

    def forward(self, points, geom_features, neighbor_idxs):
        """
        Forward pass through all residual blocks to extract features from point clouds.
        
        Args:
            points (torch.Tensor): Input point clouds.
            geom_features (torch.Tensor): Geometric features associated with points.
            neighbor_idxs (torch.Tensor): Indices of neighbors for each point.
        
        Returns:
            torch.Tensor: Extracted point features.
        """
        point_feats = torch.zeros((points.shape[0], self.dim)).to(points.device)
        for res_block in self.res_blocks:
            point_feats = res_block(point_feats, geom_features, neighbor_idxs)
        return point_feats

    def extract_features(self, points):
        """
        Utility method to extract geometric features from raw points and then process through the network.
        
        Args:
            points (torch.Tensor): Raw point cloud data.
        
        Returns:
            torch.Tensor: Extracted features from point clouds.
        """
        geom_features, neighbor_idxs = compute_geometric_features(points, self.neighbors)
        return self(points, geom_features, neighbor_idxs)

class PointCloudDecoder(nn.Module):
    """
    Decoder network for reconstructing point clouds from features.
    """
    def __init__(self, dim, hidden_dim, leaky_slope, num_output_points):
        # Initialize the PointCloudDecoder with MLP layers
        super(PointCloudDecoder, self).__init__()
        self.input_layer = MLP(dim, hidden_dim) # Input MLP
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), # First hidden linear layer
            nn.LeakyReLU(leaky_slope) # Activation function
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), # Second hidden linear layer
            nn.LeakyReLU(leaky_slope) # Activation function
        )
        self.output_layer = nn.Linear(hidden_dim, num_output_points * 3) # Output layer to produce reconstructed points

    def forward(self, point_feats):
        """
        Forward pass through the decoder to reconstruct point clouds from features.
        
        Args:
            point_feats (torch.Tensor): Features extracted from input point clouds.
        
        Returns:
            torch.Tensor: Reconstructed point cloud coordinates.
        """
        x = self.input_layer(point_feats)
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        return self.output_layer(x).reshape(-1, NUM_OUTPUT_POINTS, 3)

class StudentNetwork(nn.Module):
    """
    Student network which combines feature extraction and point cloud decoding for training.
    """
    def __init__(self, neighbors, dim, num_res_blocks):
        # Initialize the StudentNetwork with a FeatureExtractor and PointCloudDecoder
        super(StudentNetwork, self).__init__()
        self.feature_extractor = FeatureExtractor(neighbors, dim, num_res_blocks)
        self.decoder = PointCloudDecoder(dim, HIDDEN_LAYER_DIM, LEAKY_RELU_SLOPE, NUM_OUTPUT_POINTS)

    def forward(self, points):
        """
        Forward pass through the student network to extract features and reconstruct point clouds.
        
        Args:
            points (torch.Tensor): Input raw point clouds.
        
        Returns:
            tuple: Tuple containing extracted features and reconstructed point clouds.
        """
        features = self.feature_extractor.extract_features(points)
        reconstructed_points = self.decoder(features)
        return features, reconstructed_points

    def calculate_loss(self, student_features, teacher_features, mu, sigma):
        """
        Calculate the loss between features extracted by the student and those by the teacher, normalized by statistical parameters.
        
        Args:
            student_features (torch.Tensor): Features extracted by the student model.
            teacher_features (torch.Tensor): Features extracted by the teacher model, used as a target.
            mu (np.array): Mean of the features used for normalization.
            sigma (np.array): Standard deviation of the features used for normalization.
        
        Returns:
            torch.Tensor: Calculated mean squared error loss.
        """
        sigma_inv = torch.diag(torch.from_numpy(1.0 / sigma).to(teacher_features.device)) # Prepare the inverse of sigma for normalization
        teacher_features_normalized = (teacher_features - torch.from_numpy(mu).to(teacher_features.device)) @ sigma_inv
        loss = F.mse_loss(student_features, teacher_features_normalized)
        return loss
