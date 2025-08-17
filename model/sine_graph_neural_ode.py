"""
Sine Aggregation Graph Neural ODE (SineGODE) for climate dynamics

This module implements Graph Neural ODEs with sine-based message passing,
inspired by Kuramoto oscillator dynamics. The sine aggregation provides
inductive biases from coupled oscillator theory while maintaining neural
network expressiveness.

Key features:
1. Sine-based message passing inspired by Kuramoto model
2. Fully-connected graph structure for climate variable interactions
3. External and internal noise modes
4. Seasonal embedding for time-dependent behavior
5. Compatible interface with existing Neural ODE models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchdiffeq import odeint
import xarray as xr
from typing import List, Optional, Tuple

from .graph_neural_ode import SeasonalGraphEmbedding
from .sine_graph_layers import SineGraphConvBlock, KuramotoInspiredLayer


class SineGraphNeuralODEFunc(nn.Module):
    """
    Sine Graph Neural ODE function using sine-based message passing
    
    dx/dt = SineGNN(x, A, s(t)) + ε(t)
    
    Where:
    - SineGNN: Sine aggregation Graph Neural Network
    - A: Adjacency matrix (fully connected)
    - s(t): Seasonal embedding
    - ε(t): Noise (handled externally or internally)
    """
    
    def __init__(self, state_dim: int, var_names: List[str], hidden_dim: int = 64,
                 sine_type: str = 'sine', num_gnn_layers: int = 2):
        super().__init__()
        self.state_dim = state_dim
        self.var_names = var_names
        self.hidden_dim = hidden_dim
        self.sine_type = sine_type
        
        # Seasonal embedding
        self.seasonal_embedding = SeasonalGraphEmbedding(embedding_dim=8)
        
        # Create fully connected graph structure
        self.register_buffer('edge_index', self._create_fully_connected_graph(state_dim))
        
        # Input projection: combine state and seasonal features
        self.input_projection = nn.Linear(state_dim + 8, hidden_dim)
        
        # Main Sine GNN dynamics
        if sine_type == 'sine':
            self.sine_dynamics = SineGraphConvBlock(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=num_gnn_layers
            )
        elif sine_type == 'kuramoto':
            self.sine_dynamics = KuramotoInspiredLayer(
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                hidden_dim=hidden_dim
            )
        else:
            raise ValueError(f"Unknown sine_type: {sine_type}")
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, state_dim)
        
        # ENSO-specific nonlinear terms (for first two variables)
        self.enso_nonlinear = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 2)
        )
        
    def _create_fully_connected_graph(self, num_nodes: int) -> torch.Tensor:
        """Create fully connected graph edge index"""
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # No self-loops
                    edges.append([i, j])
        
        if len(edges) == 0:  # Single node case
            edges = [[0, 0]]
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute dx/dt using Sine Graph Neural Networks
        
        Args:
            t: Time tensor [1]
            x: State tensor [batch_size, state_dim]
            
        Returns:
            dx/dt tensor [batch_size, state_dim]
        """
        batch_size = x.shape[0]
        
        # Seasonal embedding
        seasonal_emb = self.seasonal_embedding(t)  # [1, 8]
        seasonal_emb = seasonal_emb.expand(batch_size, -1)  # [batch_size, 8]
        
        # Combine state and seasonal information
        x_seasonal = torch.cat([x, seasonal_emb], dim=-1)  # [batch_size, state_dim + 8]
        
        # Input projection
        x_proj = self.input_projection(x_seasonal)  # [batch_size, hidden_dim]
        
        # Process each batch element separately for sine graph operations
        batch_outputs = []
        for b in range(batch_size):
            # Single batch element: [state_dim, hidden_dim]
            x_single = x_proj[b:b+1].expand(self.state_dim, -1)  # [state_dim, hidden_dim]
            
            # Apply Sine GNN dynamics
            x_sine = self.sine_dynamics(x_single, self.edge_index)  # [state_dim, hidden_dim]
            
            # Output projection
            dxdt_single = self.output_projection(x_sine)  # [state_dim, state_dim]
            dxdt_single = dxdt_single.mean(dim=1)  # [state_dim] - average across features
            
            batch_outputs.append(dxdt_single)
        
        # Stack batch results
        dxdt = torch.stack(batch_outputs, dim=0)  # [batch_size, state_dim]
        
        # Add ENSO-specific nonlinear terms for first two variables
        if self.state_dim >= 2:
            enso_vars = x[:, :2]  # First two variables (T, H)
            enso_nonlinear = self.enso_nonlinear(enso_vars)
            dxdt[:, :2] = dxdt[:, :2] + enso_nonlinear
        
        return dxdt


class StochasticSineGraphNeuralODEFunc(SineGraphNeuralODEFunc):
    """
    Sine Graph Neural ODE function with noise directly in the vector field (internal noise)
    """
    
    def __init__(self, state_dim: int, var_names: List[str], hidden_dim: int = 64,
                 sine_type: str = 'sine', num_gnn_layers: int = 2, noise_scale: float = 0.1):
        super().__init__(state_dim, var_names, hidden_dim, sine_type, num_gnn_layers)
        self.noise_scale = noise_scale
        
        # Seasonal noise generator for time-dependent random vectors
        self.noise_generator = nn.Sequential(
            nn.Linear(8, 16),  # 8 from seasonal embedding
            nn.Tanh(),
            nn.Linear(16, state_dim),
            nn.Softplus()  # Ensure positive scaling
        )
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute dx/dt with noise directly in the vector field
        
        Args:
            t: Time tensor [1]
            x: State tensor [batch_size, state_dim]
            
        Returns:
            dx/dt tensor [batch_size, state_dim] with noise
        """
        batch_size = x.shape[0]
        
        # Get deterministic part from parent class
        dxdt_deterministic = super().forward(t, x)
        
        # Add stochastic component directly to vector field
        seasonal_emb = self.seasonal_embedding(t)  # [1, 8]
        seasonal_emb = seasonal_emb.expand(batch_size, -1)  # [batch_size, 8]
        
        # Generate seasonal noise scaling
        noise_scaling = self.noise_generator(seasonal_emb)  # [batch_size, state_dim]
        
        # Generate random noise
        noise = torch.randn_like(x) * noise_scaling * self.noise_scale
        
        return dxdt_deterministic + noise


class SineGraphNeuralODE(nn.Module):
    """
    Sine Graph Neural ODE for climate dynamics
    Uses sine-based message passing inspired by Kuramoto oscillators
    Supports both external and internal noise modes
    """
    
    def __init__(self, state_dim: int, var_names: List[str], hidden_dim: int = 64,
                 sine_type: str = 'sine', num_gnn_layers: int = 2,
                 noise_mode: str = 'external', noise_scale: float = 0.1):
        super().__init__()
        self.state_dim = state_dim
        self.var_names = var_names
        self.noise_mode = noise_mode
        self.noise_scale = noise_scale
        self.sine_type = sine_type
        
        # Create ODE function with or without internal noise
        if noise_mode == 'internal':
            self.ode_func = StochasticSineGraphNeuralODEFunc(
                state_dim, var_names, hidden_dim, sine_type, num_gnn_layers, noise_scale
            )
        else:
            self.ode_func = SineGraphNeuralODEFunc(
                state_dim, var_names, hidden_dim, sine_type, num_gnn_layers
            )
        
        # External noise model for post-integration noise
        if noise_mode == 'external':
            self.noise_net = nn.Sequential(
                nn.Linear(state_dim + 1, 32),  # +1 for seasonal component
                nn.ReLU(),
                nn.Linear(32, state_dim),
                nn.Softplus()  # Ensure positive noise scaling
            )
    
    def forward(self, x0: torch.Tensor, t: torch.Tensor, 
                add_noise: bool = False, enable_noise: bool = False) -> torch.Tensor:
        """
        Forward pass through Sine Graph Neural ODE
        
        Args:
            x0: Initial conditions [batch_size, state_dim]
            t: Time points [n_times]
            add_noise: Whether to add external noise (for external mode)
            enable_noise: Whether to enable internal noise (for internal mode)
            
        Returns:
            Solution trajectory [n_times, batch_size, state_dim]
        """
        if self.noise_mode == 'internal':
            # Internal noise: noise is part of the vector field
            if hasattr(self.ode_func, 'noise_scale'):
                original_noise_scale = self.ode_func.noise_scale
                if not enable_noise:
                    self.ode_func.noise_scale = 0.0  # Disable noise
            
            solution = odeint(self.ode_func, x0, t, method='rk4', options={'step_size': 0.1})
            
            # Restore original noise scale
            if hasattr(self.ode_func, 'noise_scale') and not enable_noise:
                self.ode_func.noise_scale = original_noise_scale
                
        else:
            # External noise: solve deterministic ODE first, then add noise
            solution = odeint(self.ode_func, x0, t, method='rk4', options={'step_size': 0.1})
            
            if add_noise:
                # Add state-dependent noise
                noise_scale = self._generate_noise(solution, t)
                noise = torch.randn_like(solution) * noise_scale
                solution = solution + noise
        
        return solution
    
    def _generate_noise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Generate state and time dependent noise scaling (external mode only)"""
        if self.noise_mode != 'external':
            raise ValueError("_generate_noise only available for external noise mode")
            
        n_times, batch_size, state_dim = x.shape
        
        noise_scales = []
        for i, time_val in enumerate(t):
            # Add seasonal component to noise
            seasonal_component = torch.sin(2 * np.pi * time_val / 12).unsqueeze(0).expand(batch_size, 1)
            x_with_season = torch.cat([x[i], seasonal_component], dim=-1)
            noise_scale = self.noise_net(x_with_season)
            noise_scales.append(noise_scale)
        
        return torch.stack(noise_scales, dim=0)
    
    def simulate(self, x0_data: xr.Dataset, nyear: int = 5, ncopy: int = 1, 
                 add_noise: bool = True, enable_noise: bool = True, device: str = 'cpu') -> xr.Dataset:
        """
        Generate simulations using the Sine Graph Neural ODE
        
        Args:
            x0_data: Initial conditions as xarray Dataset
            nyear: Number of years to simulate
            ncopy: Number of ensemble members
            add_noise: Whether to add external noise
            enable_noise: Whether to enable internal noise
            device: Device for computation
            
        Returns:
            Simulation results as xarray Dataset
        """
        self.to(device)
        self.eval()
        
        # Convert initial conditions to tensor
        x0_values = torch.stack([
            torch.tensor(x0_data[var].values.flatten()[0], dtype=torch.float32, device=device)
            for var in self.var_names
        ], dim=0)
        
        # Expand for ensemble
        x0_batch = x0_values.unsqueeze(0).expand(ncopy, -1)  # [ncopy, state_dim]
        
        # Time points (monthly)
        n_months = nyear * 12
        t = torch.linspace(0, n_months, n_months + 1, device=device)
        
        with torch.no_grad():
            # Generate simulation
            if self.noise_mode == 'internal':
                solution = self.forward(x0_batch, t, enable_noise=enable_noise)
            else:
                solution = self.forward(x0_batch, t, add_noise=add_noise)
            
            # Convert back to xarray
            solution_np = solution.detach().cpu().numpy()
            
            # Create time coordinate
            time_coord = xr.date_range(
                start='2000-01', periods=len(t), freq='MS', use_cftime=True
            )
            
            # Create dataset
            data_vars = {}
            for i, var in enumerate(self.var_names):
                data_vars[var] = xr.DataArray(
                    solution_np[:, :, i],
                    coords={'time': time_coord, 'member': np.arange(ncopy)},
                    dims=['time', 'member']
                )
            
            return xr.Dataset(data_vars)
    
    def reforecast(self, init_data: xr.Dataset, n_month: int = 21, ncopy: int = 1,
                   add_noise: bool = False, enable_noise: bool = False, device: str = 'cpu') -> xr.Dataset:
        """
        Generate reforecasts using the Sine Graph Neural ODE
        
        Args:
            init_data: Initial conditions dataset
            n_month: Forecast lead time in months
            ncopy: Number of ensemble members
            add_noise: Whether to add external noise
            enable_noise: Whether to enable internal noise
            device: Device for computation
            
        Returns:
            Forecast results as xarray Dataset
        """
        self.to(device)
        self.eval()
        
        forecasts = []
        
        for init_time in init_data.time:
            # Get initial condition
            x0_dict = {var: init_data[var].sel(time=init_time).values 
                      for var in self.var_names}
            
            x0_values = torch.stack([
                torch.tensor(x0_dict[var], dtype=torch.float32, device=device)
                for var in self.var_names
            ], dim=0)
            
            # Expand for ensemble
            x0_batch = x0_values.unsqueeze(0).expand(ncopy, -1)
            
            # Forecast time points
            t = torch.linspace(0, n_month, n_month + 1, device=device)
            
            with torch.no_grad():
                # Generate forecast
                if self.noise_mode == 'internal':
                    forecast = self.forward(x0_batch, t, enable_noise=enable_noise)
                else:
                    forecast = self.forward(x0_batch, t, add_noise=add_noise)
                forecasts.append(forecast.detach().cpu().numpy())
        
        # Combine forecasts
        forecasts_np = np.stack(forecasts, axis=1)  # [n_times, n_inits, ncopy, state_dim]
        
        # Create coordinates
        lead_coord = np.arange(n_month + 1)
        init_coord = init_data.time.values
        member_coord = np.arange(ncopy)
        
        # Create dataset
        data_vars = {}
        for i, var in enumerate(self.var_names):
            data_vars[var] = xr.DataArray(
                forecasts_np[:, :, :, i],
                coords={'lead': lead_coord, 'init_time': init_coord, 'member': member_coord},
                dims=['lead', 'init_time', 'member']
            )
        
        return xr.Dataset(data_vars)
