"""
Spatial-Temporal Graph Neural Network for Visibility Prediction
================================================================

This module implements a Spatial-Temporal GNN that explicitly models both
spatial relationships between stations and temporal dependencies in visibility data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for capturing time-varying patterns.
    
    Args:
        hidden_dim: Dimension of hidden features
        num_heads: Number of attention heads
    """
    
    def __init__(self, hidden_dim, num_heads=4):
        super(TemporalAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
        Returns:
            Attention output [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention calculation
        Q = Q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(attn_output)
        
        return output


class SpatialGraphConv(nn.Module):
    """
    Spatial graph convolution layer with learnable edge weights.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
    """
    
    def __init__(self, in_features, out_features):
        super(SpatialGraphConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.edge_weight_net = nn.Sequential(
            nn.Linear(2 * in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [batch_size, num_nodes, in_features]
            edge_index: Edge connectivity [2, num_edges] or adjacency matrix [num_nodes, num_nodes]
        Returns:
            Updated node features [batch_size, num_nodes, out_features]
        """
        batch_size, num_nodes, in_features = x.shape
        
        # Transform features
        x_transformed = self.linear(x)
        
        # If edge_index is an adjacency matrix, convert to edge list
        if len(edge_index.shape) == 2 and edge_index.shape[0] == edge_index.shape[1]:
            # It's an adjacency matrix - use it directly
            # Aggregate using matrix multiplication
            # edge_index: [num_nodes, num_nodes]
            # x_transformed: [batch_size, num_nodes, out_features]
            aggregated = torch.matmul(edge_index.unsqueeze(0), x_transformed)
            return aggregated
        
        # Otherwise, use edge list format
        aggregated = torch.zeros(batch_size, num_nodes, x_transformed.shape[-1], 
                                device=x.device)
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            # Compute edge weight based on node features
            edge_feat = torch.cat([x[:, src, :], x[:, dst, :]], dim=-1)
            edge_weight = self.edge_weight_net(edge_feat)
            
            # Weighted aggregation
            aggregated[:, dst, :] += edge_weight * x_transformed[:, src, :]
        
        return aggregated


class STGNNVisibility(nn.Module):
    """
    Spatial-Temporal Graph Neural Network for visibility prediction.
    
    This advanced model combines spatial graph convolutions with temporal
    attention mechanisms to capture complex spatio-temporal patterns in
    multi-station visibility data.
    
    Args:
        num_stations: Number of weather stations
        input_dim: Dimension of input features per station
        hidden_dim: Dimension of hidden layers
        output_dim: Dimension of output (prediction horizon)
        num_spatial_layers: Number of spatial graph convolution layers
        num_temporal_layers: Number of temporal attention layers
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(self, num_stations=10, input_dim=7, hidden_dim=128, 
                 output_dim=1, num_spatial_layers=2, num_temporal_layers=2,
                 num_heads=4, dropout=0.1):
        super(STGNNVisibility, self).__init__()
        
        self.num_stations = num_stations
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Spatial layers
        self.spatial_layers = nn.ModuleList([
            SpatialGraphConv(hidden_dim, hidden_dim)
            for _ in range(num_spatial_layers)
        ])
        
        self.spatial_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_spatial_layers)
        ])
        
        # Temporal attention layers
        self.temporal_layers = nn.ModuleList([
            TemporalAttention(hidden_dim, num_heads)
            for _ in range(num_temporal_layers)
        ])
        
        self.temporal_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_temporal_layers)
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Learnable edge index (fully connected graph)
        self.register_buffer('edge_index', self._create_edge_index(num_stations))
        
    def _create_edge_index(self, num_nodes):
        """Create a fully connected graph edge index."""
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
        # Return as long tensor
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
    def forward(self, x, adj_matrix=None):
        """
        Forward pass for spatio-temporal visibility prediction.
        
        Args:
            x: Input features [batch_size, seq_len, num_stations, input_dim]
            adj_matrix: Adjacency matrix [batch_size, num_stations, num_stations] or [num_stations, num_stations]
                       If None, uses default fully connected graph
            
        Returns:
            Visibility predictions [batch_size, num_stations, output_dim]
        """
        batch_size, seq_len, num_stations, _ = x.shape
        
        # Handle adjacency matrix
        if adj_matrix is None:
            # Create default adjacency (fully connected)
            adj_matrix = torch.ones(num_stations, num_stations, device=x.device)
            adj_matrix.fill_diagonal_(0)
            adj_matrix = adj_matrix / (adj_matrix.sum(dim=1, keepdim=True) + 1e-6)
        elif len(adj_matrix.shape) == 3:
            # Batch of adjacency matrices - use first one
            adj_matrix = adj_matrix[0]
        
        # Embed input features
        x = x.reshape(batch_size * seq_len, num_stations, self.input_dim)
        x = self.input_embedding(x)  # [batch_size * seq_len, num_stations, hidden_dim]
        
        # Spatial processing
        spatial_features = x
        for spatial_layer, spatial_norm in zip(self.spatial_layers, self.spatial_norms):
            spatial_out = spatial_layer(spatial_features, adj_matrix)
            spatial_features = spatial_norm(spatial_out + spatial_features)
        
        # Reshape for temporal processing
        spatial_features = spatial_features.reshape(batch_size, seq_len, num_stations, self.hidden_dim)
        
        # Temporal processing for each station
        temporal_features_list = []
        for i in range(num_stations):
            station_seq = spatial_features[:, :, i, :]  # [batch_size, seq_len, hidden_dim]
            
            temp_feat = station_seq
            for temporal_layer, temporal_norm in zip(self.temporal_layers, self.temporal_norms):
                temp_out = temporal_layer(temp_feat)
                temp_feat = temporal_norm(temp_out + temp_feat)
            
            temporal_features_list.append(temp_feat[:, -1, :])  # Use last timestep
        
        temporal_features = torch.stack(temporal_features_list, dim=1)  # [batch_size, num_stations, hidden_dim]
        
        # Combine spatial and temporal features
        spatial_final = spatial_features[:, -1, :, :]  # [batch_size, num_stations, hidden_dim]
        combined = torch.cat([spatial_final, temporal_features], dim=-1)
        fused_features = self.fusion(combined)
        
        # Generate predictions
        predictions = self.output_proj(fused_features)  # [batch_size, num_stations, output_dim]
        
        return predictions
