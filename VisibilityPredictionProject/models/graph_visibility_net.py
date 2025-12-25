"""
Graph Neural Network for Multi-Station Visibility Prediction
============================================================

This module implements a Graph Neural Network (GNN) architecture for predicting
visibility across multiple weather stations. The model captures spatial 
relationships between stations using graph convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """
    Graph Convolution Layer for processing station-level features.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        dropout: Dropout rate for regularization
    """
    
    def __init__(self, in_features, out_features, dropout=0.1):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x, adj_matrix):
        """
        Forward pass of graph convolution.
        
        Args:
            x: Node features [batch_size, num_nodes, in_features]
            adj_matrix: Adjacency matrix [num_nodes, num_nodes]
            
        Returns:
            Updated node features [batch_size, num_nodes, out_features]
        """
        # Linear transformation
        x = self.linear(x)
        
        # Graph convolution: aggregate neighbor information
        # adj_matrix shape: [num_nodes, num_nodes]
        # x shape: [batch_size, num_nodes, out_features]
        x = torch.matmul(adj_matrix.unsqueeze(0), x)
        
        # Apply activation and dropout
        x = self.activation(x)
        x = self.dropout(x)
        
        return x


class GraphVisibilityNet(nn.Module):
    """
    GNN-based model for multi-station visibility prediction.
    
    This model uses multiple graph convolution layers to capture spatial
    dependencies between weather stations and temporal patterns in the data.
    
    Args:
        num_stations: Number of weather stations
        input_dim: Dimension of input features per station
        hidden_dim: Dimension of hidden layers
        output_dim: Dimension of output (prediction horizon)
        num_layers: Number of graph convolution layers
        dropout: Dropout rate
    """
    
    def __init__(self, num_stations=10, input_dim=7, hidden_dim=64, 
                 output_dim=1, num_layers=3, dropout=0.1):
        super(GraphVisibilityNet, self).__init__()
        
        self.num_stations = num_stations
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Graph convolution layers
        self.graph_convs = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Temporal encoding using LSTM
        self.temporal_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Learnable adjacency matrix
        self.adj_matrix = nn.Parameter(torch.randn(num_stations, num_stations))
        
    def forward(self, x, adj_matrix=None):
        """
        Forward pass for visibility prediction.
        
        Args:
            x: Input features [batch_size, seq_len, num_stations, input_dim]
            adj_matrix: Optional adjacency matrix [num_stations, num_stations]
                       If None, uses learnable adjacency matrix
            
        Returns:
            Visibility predictions [batch_size, num_stations, output_dim]
        """
        batch_size, seq_len, num_stations, _ = x.shape
        
        # Use provided adjacency matrix or learnable one
        if adj_matrix is None:
            adj_matrix = torch.softmax(self.adj_matrix, dim=-1)
        
        # Project input
        x = x.reshape(batch_size * seq_len, num_stations, self.input_dim)
        x = self.input_proj(x)  # [batch_size * seq_len, num_stations, hidden_dim]
        
        # Apply graph convolutions
        for graph_conv in self.graph_convs:
            x_residual = x
            x = graph_conv(x, adj_matrix)
            x = x + x_residual  # Residual connection
        
        # Reshape for temporal encoding
        x = x.reshape(batch_size, seq_len, num_stations, self.hidden_dim)
        
        # Process each station's temporal sequence
        outputs = []
        for i in range(num_stations):
            station_features = x[:, :, i, :]  # [batch_size, seq_len, hidden_dim]
            temporal_out, _ = self.temporal_encoder(station_features)
            # Use last timestep
            station_pred = self.output_proj(temporal_out[:, -1, :])  # [batch_size, output_dim]
            outputs.append(station_pred)
        
        # Stack predictions for all stations
        outputs = torch.stack(outputs, dim=1)  # [batch_size, num_stations, output_dim]
        
        return outputs
    
    def get_adjacency_matrix(self):
        """
        Get the learned adjacency matrix.
        
        Returns:
            Adjacency matrix representing station relationships
        """
        return torch.softmax(self.adj_matrix, dim=-1).detach()
