import torch
import torch.nn as nn
import torch.nn.functional as F
from models_utils import * # Import utility functions and classes such as MLP and compute_geometric_features


# Define constants
HIDDEN_LAYER_DIM = 128  # Number of dimensions in hidden layers of MLPs
LEAKY_RELU_SLOPE = 0.2  # Negative slope coefficient for LeakyReLU activations
NUM_OUTPUT_POINTS = 2048  # Number of output points for the decoder

class MLP(nn.Module):
    """
    A simple multilayer perceptron (MLP) module with one linear layer followed by a LeakyReLU activation.
    """
    def __init__(self, input_dim, output_dim):
        # Initialize the MLP with a single linear layer followed by a LeakyReLU activation
        super(MLP, self).__init__()
        self.dense = nn.Linear(input_dim, output_dim) # Linear layer
        self.activation_fn = nn.LeakyReLU(LEAKY_RELU_SLOPE) # LeakyReLU activation function

    def forward(self, x):
        """
        Forward pass of the MLP.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after applying linear layer and activation function.
        """
        x = self.activation_fn(self.dense(x))
        return x

class ResidualBlock(nn.Module):
    """
    A residual block that uses MLPs for transformation and Local Feature Aggregation (LFA).
    """
    def __init__(self, dim):
        # Initialize a ResidualBlock with MLPs and LFAs
        super(ResidualBlock, self).__init__()
        self.mlp_initial = MLP(dim, dim // 4) # Initial transformation reducing dimensionality
        self.lfa1 = LFA(dim // 4) # First local feature aggregation
        self.lfa2 = LFA(dim // 2) # Second local feature aggregation
        self.mlp_final = MLP(dim, dim) # Final MLP to transform back to original dimensionality
        self.mlp_residual = MLP(dim, dim) # MLP for processing the residual connection

    def forward(self, input_features, geom_features, closest_indices):
        """
        Forward pass through the residual block, combining MLP outputs and a residual connection.
        
        Args:
            input_features (torch.Tensor): Input feature tensor.
            geom_features (torch.Tensor): Geometric features from point cloud.
            closest_indices (torch.Tensor): Indices of closest points used for feature aggregation.
        
        Returns:
            torch.Tensor: Output features after residual connection.
        """
        residual = self.mlp_residual(input_features)
        activated = self.mlp_initial(input_features)
        activated = self.lfa1(activated, geom_features, closest_indices)
        activated = self.lfa2(activated, geom_features, closest_indices)
        activated = self.mlp_final(activated)
        return activated + residual

class LFA(nn.Module):
    """
    Local Feature Aggregation (LFA) module which aggregates features from nearest neighbors.
    """
    def __init__(self, d_lfa):
        # Initialize the LFA with an MLP
        super(LFA, self).__init__()
        self.mlp = MLP(4, d_lfa) # MLP for processing aggregated features

    def forward(self, input_features, geom_features, closest_indices):
        """
        Forward pass of LFA, aggregating features from input and geometric features based on closest indices.
        
        Args:
            input_features (torch.Tensor): Input feature tensor.
            geom_features (torch.Tensor): Geometric features tensor.
            closest_indices (torch.Tensor): Indices of closest points for aggregation.
        
        Returns:
            torch.Tensor: Aggregated output features.
        """
        geom_outputs = self.mlp(geom_features)
        relevant_input_features = input_features[closest_indices]
        combined_features = torch.cat((geom_outputs, relevant_input_features), dim=-1)
        output_features = torch.mean(combined_features, dim=1)
        return output_features

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
