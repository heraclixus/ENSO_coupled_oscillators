"""
Sine Aggregation Graph Layers inspired by Kuramoto Oscillator dynamics

This module implements graph neural network layers that use sine-based message passing,
inspired by the Kuramoto model of coupled oscillators. The key idea is to use
sine functions for message aggregation while maintaining neural network expressiveness.

Key concepts from Kuramoto model:
- Phase coupling: sin(θ_j - θ_i) terms
- Frequency heterogeneity: Different natural frequencies
- Collective synchronization behavior
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import Optional, Tuple


class SineAggregationLayer(MessagePassing):
    """
    Sine-based message passing layer inspired by Kuramoto oscillators
    
    The message passing follows:
    m_{ij} = sin(NN_msg(x_j) - NN_phase(x_i)) * NN_weight(x_i, x_j)
    
    Where:
    - NN_msg: Neural network producing "phase-like" messages
    - NN_phase: Neural network producing "phase-like" node states  
    - NN_weight: Neural network producing coupling weights
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__(aggr='mean')  # Use mean aggregation after sine transformation
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Neural networks for phase-like representations
        self.phase_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.message_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Neural network for coupling weights
        self.weight_net = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Ensure positive coupling weights
        )
        
        # Output transformation
        self.output_net = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sine-based message passing
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            
        Returns:
            Updated node features [num_nodes, output_dim]
        """
        # Compute phase-like representations
        node_phases = self.phase_net(x)  # [num_nodes, output_dim]
        
        # Propagate messages
        out = self.propagate(edge_index, x=x, node_phases=node_phases)
        
        # Apply output transformation
        out = self.output_net(out)
        
        return out
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, 
                node_phases_i: torch.Tensor, node_phases_j: torch.Tensor) -> torch.Tensor:
        """
        Compute messages using sine-based coupling
        
        Args:
            x_i: Source node features [num_edges, input_dim]
            x_j: Target node features [num_edges, input_dim]
            node_phases_i: Source node phases [num_edges, output_dim]
            node_phases_j: Target node phases [num_edges, output_dim]
            
        Returns:
            Messages [num_edges, output_dim]
        """
        # Compute message phases
        message_phases = self.message_net(x_j)  # [num_edges, output_dim]
        
        # Compute coupling weights
        edge_features = torch.cat([x_i, x_j], dim=-1)  # [num_edges, 2*input_dim]
        coupling_weights = self.weight_net(edge_features)  # [num_edges, output_dim]
        
        # Sine-based message: sin(message_phase - node_phase) * coupling_weight
        phase_diff = message_phases - node_phases_i  # [num_edges, output_dim]
        sine_messages = torch.sin(phase_diff)  # [num_edges, output_dim]
        
        # Weight the sine messages
        weighted_messages = coupling_weights * sine_messages  # [num_edges, output_dim]
        
        return weighted_messages


class SineGraphConvBlock(nn.Module):
    """
    Multi-layer sine aggregation block with residual connections
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer
        self.layers.append(SineAggregationLayer(input_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(SineAggregationLayer(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.layers.append(SineAggregationLayer(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection projection if needed
        self.residual_proj = None
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through sine aggregation layers
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            
        Returns:
            Updated node features [num_nodes, output_dim]
        """
        residual = x
        
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x, edge_index)
            if i < len(self.norms):
                x = norm(x)
                x = F.relu(x)
                x = self.dropout(x)
        
        # Final layer without activation if we have multiple layers
        if len(self.layers) > len(self.norms):
            x = self.layers[-1](x, edge_index)
        
        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        
        if x.shape == residual.shape:
            x = x + residual
        
        return x


class KuramotoInspiredLayer(nn.Module):
    """
    Kuramoto-inspired layer with explicit frequency and phase dynamics
    
    This layer more directly implements Kuramoto-like dynamics:
    dθ_i/dt = ω_i + Σ_j K_ij * sin(θ_j - θ_i)
    
    Where:
    - θ_i: Phase of oscillator i (learned from node features)
    - ω_i: Natural frequency of oscillator i (learned parameter)
    - K_ij: Coupling strength between oscillators i and j (learned)
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Networks to compute phases and frequencies from node features
        self.phase_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.frequency_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Network to compute coupling matrix
        self.coupling_net = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Output transformation
        self.output_net = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing Kuramoto-like dynamics
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            
        Returns:
            Phase derivatives (representing dx/dt) [num_nodes, output_dim]
        """
        num_nodes = x.shape[0]
        
        # Compute phases and natural frequencies
        phases = self.phase_net(x)  # [num_nodes, output_dim]
        frequencies = self.frequency_net(x)  # [num_nodes, output_dim]
        
        # Initialize phase derivatives with natural frequencies
        phase_derivatives = frequencies.clone()  # [num_nodes, output_dim]
        
        # Compute coupling terms
        source_nodes, target_nodes = edge_index
        
        for i in range(num_nodes):
            # Find neighbors of node i
            neighbor_mask = target_nodes == i
            if neighbor_mask.sum() == 0:
                continue
                
            neighbors = source_nodes[neighbor_mask]
            
            # Compute coupling contributions
            for j in neighbors:
                # Coupling strength K_ij
                edge_features = torch.cat([x[i:i+1], x[j:j+1]], dim=-1)
                coupling_strength = self.coupling_net(edge_features)  # [1, 1]
                
                # Phase difference: θ_j - θ_i
                phase_diff = phases[j] - phases[i]  # [output_dim]
                
                # Coupling term: K_ij * sin(θ_j - θ_i)
                coupling_term = coupling_strength.squeeze() * torch.sin(phase_diff)
                
                # Add to phase derivative
                phase_derivatives[i] += coupling_term
        
        # Apply output transformation
        output = self.output_net(phase_derivatives)
        
        return output


def create_sine_aggregation_block(input_dim: int, hidden_dim: int, output_dim: int,
                                 aggregation_type: str = 'sine', num_layers: int = 2) -> nn.Module:
    """
    Factory function to create different types of sine aggregation blocks
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        output_dim: Output feature dimension
        aggregation_type: Type of sine aggregation ('sine', 'kuramoto')
        num_layers: Number of layers
        
    Returns:
        Sine aggregation block
    """
    if aggregation_type == 'sine':
        return SineGraphConvBlock(input_dim, hidden_dim, output_dim, num_layers)
    elif aggregation_type == 'kuramoto':
        # For Kuramoto, we use a single layer approach
        return KuramotoInspiredLayer(input_dim, output_dim, hidden_dim)
    else:
        raise ValueError(f"Unknown aggregation type: {aggregation_type}")
