"""
Physics-Informed Graph Neural ODE (PhyGODE) for climate dynamics

This module implements a physics-informed Graph Neural ODE that combines
the structured approach of PhysicsInformedODE with Graph Neural Networks.

Key features:
1. Seasonal linear operator with learnable Fourier coefficients
2. Graph-based nonlinear terms using GCN/GAT
3. ENSO-specific physics constraints
4. Fully-connected graph structure
5. External and internal noise modes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchdiffeq import odeint
import xarray as xr
from typing import List, Optional, Tuple
from torch_geometric.nn import GCNConv, GATConv

from .graph_neural_ode import SeasonalGraphEmbedding, GraphConvBlock


class SeasonalLinearGraphOperator(nn.Module):
    """
    Seasonal linear operator using Fourier decomposition (like PhysicsInformedODE)
    L(t) = L₀ + Σₖ [Lₖᶜcos(kωt) + Lₖˢsin(kωt)]
    """
    
    def __init__(self, state_dim: int, fourier_modes: int = 2):
        super().__init__()
        self.state_dim = state_dim
        self.fourier_modes = fourier_modes
        self.omega = 2 * np.pi / 12  # Monthly frequency
        
        # Fourier coefficients: [state_dim, state_dim, 2*fourier_modes + 1]
        # Index 0: L₀ (annual mean)
        # Index 1,2: L₁ᶜ, L₁ˢ (annual cycle)
        # Index 3,4: L₂ᶜ, L₂ˢ (semi-annual cycle), etc.
        n_coeffs = 2 * fourier_modes + 1
        self.fourier_coeffs = nn.Parameter(torch.randn(state_dim, state_dim, n_coeffs) * 0.1)
        
    def forward(self, t: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Compute L(t) for given time
        
        Args:
            t: Time tensor [1]
            batch_size: Batch size for broadcasting
            
        Returns:
            Linear operator L(t) [batch_size, state_dim, state_dim]
        """
        device = self.fourier_coeffs.device
        
        # Start with annual mean (L₀)
        L = self.fourier_coeffs[:, :, 0].unsqueeze(0).expand(batch_size, -1, -1).clone()
        
        # Add Fourier components
        for k in range(1, self.fourier_modes + 1):
            cos_term = torch.cos(k * self.omega * t)
            sin_term = torch.sin(k * self.omega * t)
            
            # Add cosine term (Lₖᶜ)
            L += self.fourier_coeffs[:, :, 2*k-1].unsqueeze(0) * cos_term
            
            # Add sine term (Lₖˢ)
            L += self.fourier_coeffs[:, :, 2*k].unsqueeze(0) * sin_term
        
        return L


class GraphNonlinearTerms(nn.Module):
    """
    Graph-based nonlinear terms using GNN layers
    Combines general nonlinear terms with ENSO-specific physics
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 64, gnn_type: str = 'gcn'):
        super().__init__()
        self.state_dim = state_dim
        self.gnn_type = gnn_type
        
        # Create fully connected graph structure
        self.register_buffer('edge_index', self._create_fully_connected_graph(state_dim))
        
        # Quadratic terms network (x²)
        self.quadratic_graph = GraphConvBlock(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            output_dim=state_dim,
            gnn_type=gnn_type,
            num_layers=2
        )
        
        # Cubic terms network (x³)
        self.cubic_graph = GraphConvBlock(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            output_dim=state_dim,
            gnn_type=gnn_type,
            num_layers=2
        )
        
        # ENSO-specific nonlinear terms (for first two variables: T, H)
        if state_dim >= 2:
            # T equation: includes T², TH, T³, T²H terms
            self.enso_T_graph = GraphConvBlock(
                input_dim=5,  # [T, H, T², TH, T³]
                hidden_dim=32,
                output_dim=1,
                gnn_type=gnn_type,
                num_layers=2
            )
            
            # H equation: includes T², TH, TH² terms  
            self.enso_H_graph = GraphConvBlock(
                input_dim=5,  # [T, H, T², TH, TH²]
                hidden_dim=32,
                output_dim=1,
                gnn_type=gnn_type,
                num_layers=2
            )
            
            # Create edge indices for ENSO terms (smaller graphs)
            self.register_buffer('enso_edge_index', self._create_fully_connected_graph(5))
    
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute nonlinear terms N(x) using graph networks
        
        Args:
            x: State tensor [batch_size, state_dim]
            
        Returns:
            Nonlinear terms [batch_size, state_dim]
        """
        batch_size = x.shape[0]
        
        # Process each batch element separately for graph operations
        batch_quad_terms = []
        batch_cubic_terms = []
        
        for b in range(batch_size):
            # Single batch element: [state_dim, 1] -> [state_dim, state_dim] for graph input
            x_single = x[b:b+1].expand(self.state_dim, -1)  # [state_dim, state_dim]
            
            # Quadratic terms
            quad_single = self.quadratic_graph(x_single, self.edge_index)  # [state_dim, state_dim]
            quad_single = quad_single.mean(dim=1)  # [state_dim] - average across features
            batch_quad_terms.append(quad_single)
            
            # Cubic terms
            cubic_single = self.cubic_graph(x_single, self.edge_index)  # [state_dim, state_dim]
            cubic_single = cubic_single.mean(dim=1)  # [state_dim] - average across features
            batch_cubic_terms.append(cubic_single)
        
        quad_terms = torch.stack(batch_quad_terms, dim=0)  # [batch_size, state_dim]
        cubic_terms = torch.stack(batch_cubic_terms, dim=0)  # [batch_size, state_dim]
        
        # Start with general terms
        nonlinear_terms = quad_terms + cubic_terms
        
        # Add ENSO-specific terms for first two variables
        if self.state_dim >= 2:
            T = x[:, 0:1]  # [batch_size, 1]
            H = x[:, 1:2]  # [batch_size, 1]
            
            # ENSO nonlinear features
            T2 = T * T
            TH = T * H
            T3 = T * T * T
            TH2 = T * H * H
            
            # Process ENSO terms for each batch element
            batch_enso_T_contrib = []
            batch_enso_H_contrib = []
            
            for b in range(batch_size):
                # T equation features: [T, H, T², TH, T³]
                enso_T_features = torch.cat([T[b:b+1], H[b:b+1], T2[b:b+1], TH[b:b+1], T3[b:b+1]], dim=1)  # [1, 5]
                enso_T_input = enso_T_features.expand(5, -1)  # [5, 5] for graph input
                
                enso_T_single = self.enso_T_graph(enso_T_input, self.enso_edge_index)  # [5, 1]
                enso_T_contrib_single = enso_T_single.mean(dim=0)  # [1]
                batch_enso_T_contrib.append(enso_T_contrib_single)
                
                # H equation features: [T, H, T², TH, TH²]
                enso_H_features = torch.cat([T[b:b+1], H[b:b+1], T2[b:b+1], TH[b:b+1], TH2[b:b+1]], dim=1)  # [1, 5]
                enso_H_input = enso_H_features.expand(5, -1)  # [5, 5] for graph input
                
                enso_H_single = self.enso_H_graph(enso_H_input, self.enso_edge_index)  # [5, 1]
                enso_H_contrib_single = enso_H_single.mean(dim=0)  # [1]
                batch_enso_H_contrib.append(enso_H_contrib_single)
            
            enso_T_contrib = torch.stack(batch_enso_T_contrib, dim=0)  # [batch_size, 1]
            enso_H_contrib = torch.stack(batch_enso_H_contrib, dim=0)  # [batch_size, 1]
            
            # Add ENSO contributions to first two variables
            nonlinear_terms[:, 0:1] += enso_T_contrib
            nonlinear_terms[:, 1:2] += enso_H_contrib
        
        return nonlinear_terms
    



class GraphNoiseModel(nn.Module):
    """
    Graph-based noise model with seasonal and state dependence
    """
    
    def __init__(self, state_dim: int, gnn_type: str = 'gcn'):
        super().__init__()
        self.state_dim = state_dim
        
        # Seasonal noise amplitude (12 months)
        self.seasonal_noise_log_amp = nn.Parameter(torch.zeros(state_dim, 12))
        
        # State-dependent noise scaling using graph network
        self.register_buffer('edge_index', self._create_fully_connected_graph(state_dim))
        
        self.noise_graph = GraphConvBlock(
            input_dim=state_dim + 1,  # state + seasonal component
            hidden_dim=32,
            output_dim=state_dim,
            gnn_type=gnn_type,
            num_layers=2
        )
        
    def _create_fully_connected_graph(self, num_nodes: int) -> torch.Tensor:
        """Create fully connected graph edge index"""
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edges.append([i, j])
        
        if len(edges) == 0:
            edges = [[0, 0]]
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Generate noise scaling
        
        Args:
            x: State tensor [batch_size, state_dim]
            t: Time tensor [1]
            
        Returns:
            Noise scaling [batch_size, state_dim]
        """
        batch_size = x.shape[0]
        
        # Seasonal component
        month = (t % 12).long()
        seasonal_amp = torch.exp(self.seasonal_noise_log_amp[:, month]).squeeze()  # [state_dim]
        seasonal_amp = seasonal_amp.unsqueeze(0).expand(batch_size, -1)  # [batch_size, state_dim]
        
        # Seasonal embedding for graph input
        seasonal_emb = torch.sin(2 * np.pi * t / 12).expand(batch_size, 1)  # [batch_size, 1]
        
        # Process each batch element separately
        batch_state_scaling = []
        
        for b in range(batch_size):
            # Combine state and seasonal information for single batch
            graph_input_single = torch.cat([x[b:b+1], seasonal_emb[b:b+1]], dim=-1)  # [1, state_dim + 1]
            graph_input_expanded = graph_input_single.expand(self.state_dim + 1, -1)  # [state_dim + 1, state_dim + 1]
            
            # State-dependent scaling
            state_scaling_single = self.noise_graph(graph_input_expanded, self.edge_index)  # [state_dim + 1, state_dim]
            state_scaling_single = state_scaling_single[:self.state_dim].mean(dim=1)  # [state_dim] - take first state_dim nodes
            state_scaling_single = torch.softplus(state_scaling_single)  # Ensure positive
            
            batch_state_scaling.append(state_scaling_single)
        
        state_scaling = torch.stack(batch_state_scaling, dim=0)  # [batch_size, state_dim]
        
        return seasonal_amp * state_scaling
    



class PhysicsGraphNeuralODEFunc(nn.Module):
    """
    Physics-informed Graph Neural ODE function
    
    dx/dt = L(t) * x + N_graph(x) + ε(t)
    
    Where:
    - L(t): Seasonal linear operator with Fourier decomposition
    - N_graph(x): Graph-based nonlinear terms
    - ε(t): Noise (handled externally or internally)
    """
    
    def __init__(self, state_dim: int, var_names: List[str], hidden_dim: int = 64,
                 gnn_type: str = 'gcn', fourier_modes: int = 2):
        super().__init__()
        self.state_dim = state_dim
        self.var_names = var_names
        self.hidden_dim = hidden_dim
        self.gnn_type = gnn_type
        
        # Seasonal linear operator
        self.linear_operator = SeasonalLinearGraphOperator(state_dim, fourier_modes)
        
        # Graph-based nonlinear terms
        self.nonlinear_terms = GraphNonlinearTerms(state_dim, hidden_dim, gnn_type)
        
        # Graph-based noise model
        self.noise_model = GraphNoiseModel(state_dim, gnn_type)
        
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute dx/dt using physics-informed graph networks
        
        Args:
            t: Time tensor [1]
            x: State tensor [batch_size, state_dim]
            
        Returns:
            dx/dt tensor [batch_size, state_dim]
        """
        batch_size = x.shape[0]
        
        # Linear dynamics: L(t) * x
        L_t = self.linear_operator(t, batch_size)  # [batch_size, state_dim, state_dim]
        linear_term = torch.bmm(L_t, x.unsqueeze(-1)).squeeze(-1)  # [batch_size, state_dim]
        
        # Nonlinear dynamics: N_graph(x)
        nonlinear_term = self.nonlinear_terms(x)  # [batch_size, state_dim]
        
        # Combine terms
        dxdt = linear_term + nonlinear_term
        
        return dxdt


class StochasticPhysicsGraphNeuralODEFunc(PhysicsGraphNeuralODEFunc):
    """
    Physics-informed Graph Neural ODE function with internal noise
    """
    
    def __init__(self, state_dim: int, var_names: List[str], hidden_dim: int = 64,
                 gnn_type: str = 'gcn', fourier_modes: int = 2, noise_scale: float = 0.1):
        super().__init__(state_dim, var_names, hidden_dim, gnn_type, fourier_modes)
        self.noise_scale = noise_scale
        
        # Seasonal embedding for noise
        self.seasonal_embedding = SeasonalGraphEmbedding(embedding_dim=8)
        
        # Seasonal noise generator
        self.noise_generator = nn.Sequential(
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.Linear(16, state_dim),
            nn.Softplus()
        )
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute dx/dt with internal noise
        
        Args:
            t: Time tensor [1]
            x: State tensor [batch_size, state_dim]
            
        Returns:
            dx/dt tensor [batch_size, state_dim] with noise
        """
        batch_size = x.shape[0]
        
        # Get deterministic part
        dxdt_deterministic = super().forward(t, x)
        
        # Add stochastic component
        seasonal_emb = self.seasonal_embedding(t)  # [1, 8]
        seasonal_emb = seasonal_emb.expand(batch_size, -1)  # [batch_size, 8]
        
        # Generate seasonal noise scaling
        noise_scaling = self.noise_generator(seasonal_emb)  # [batch_size, state_dim]
        
        # Generate random noise
        noise = torch.randn_like(x) * noise_scaling * self.noise_scale
        
        return dxdt_deterministic + noise


class PhysicsGraphNeuralODE(nn.Module):
    """
    Physics-informed Graph Neural ODE for climate dynamics
    Combines structured physics (like PhysicsInformedODE) with Graph Neural Networks
    """
    
    def __init__(self, state_dim: int, var_names: List[str], hidden_dim: int = 64,
                 gnn_type: str = 'gcn', fourier_modes: int = 2,
                 noise_mode: str = 'external', noise_scale: float = 0.1):
        super().__init__()
        self.state_dim = state_dim
        self.var_names = var_names
        self.noise_mode = noise_mode
        self.noise_scale = noise_scale
        self.gnn_type = gnn_type
        
        # Create ODE function with or without internal noise
        if noise_mode == 'internal':
            self.ode_func = StochasticPhysicsGraphNeuralODEFunc(
                state_dim, var_names, hidden_dim, gnn_type, fourier_modes, noise_scale
            )
        else:
            self.ode_func = PhysicsGraphNeuralODEFunc(
                state_dim, var_names, hidden_dim, gnn_type, fourier_modes
            )
        
        # External noise model
        if noise_mode == 'external':
            self.noise_net = nn.Sequential(
                nn.Linear(state_dim + 1, 32),
                nn.ReLU(),
                nn.Linear(32, state_dim),
                nn.Softplus()
            )
    
    def forward(self, x0: torch.Tensor, t: torch.Tensor,
                add_noise: bool = False, enable_noise: bool = False) -> torch.Tensor:
        """Forward pass through Physics Graph Neural ODE"""
        if self.noise_mode == 'internal':
            # Internal noise handling
            if hasattr(self.ode_func, 'noise_scale'):
                original_noise_scale = self.ode_func.noise_scale
                if not enable_noise:
                    self.ode_func.noise_scale = 0.0
            
            solution = odeint(self.ode_func, x0, t, method='rk4', options={'step_size': 0.1})
            
            if hasattr(self.ode_func, 'noise_scale') and not enable_noise:
                self.ode_func.noise_scale = original_noise_scale
        else:
            # External noise handling
            solution = odeint(self.ode_func, x0, t, method='rk4', options={'step_size': 0.1})
            
            if add_noise:
                noise_scale = self._generate_noise(solution, t)
                noise = torch.randn_like(solution) * noise_scale
                solution = solution + noise
        
        return solution
    
    def _generate_noise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Generate external noise scaling"""
        if self.noise_mode != 'external':
            raise ValueError("_generate_noise only available for external noise mode")
            
        n_times, batch_size, state_dim = x.shape
        noise_scales = []
        
        for i, time_val in enumerate(t):
            seasonal_component = torch.sin(2 * np.pi * time_val / 12).unsqueeze(0).expand(batch_size, 1)
            x_with_season = torch.cat([x[i], seasonal_component], dim=-1)
            noise_scale = self.noise_net(x_with_season)
            noise_scales.append(noise_scale)
        
        return torch.stack(noise_scales, dim=0)
    
    def simulate(self, x0_data: xr.Dataset, nyear: int = 5, ncopy: int = 1,
                 add_noise: bool = True, enable_noise: bool = True, device: str = 'cpu') -> xr.Dataset:
        """Generate simulations (same interface as other models)"""
        self.to(device)
        self.eval()
        
        # Convert initial conditions
        x0_values = torch.stack([
            torch.tensor(x0_data[var].values.flatten()[0], dtype=torch.float32, device=device)
            for var in self.var_names
        ], dim=0)
        
        x0_batch = x0_values.unsqueeze(0).expand(ncopy, -1)
        n_months = nyear * 12
        t = torch.linspace(0, n_months, n_months + 1, device=device)
        
        with torch.no_grad():
            if self.noise_mode == 'internal':
                solution = self.forward(x0_batch, t, enable_noise=enable_noise)
            else:
                solution = self.forward(x0_batch, t, add_noise=add_noise)
            
            solution_np = solution.detach().cpu().numpy()
            time_coord = xr.date_range(start='2000-01', periods=len(t), freq='MS', use_cftime=True)
            
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
        """Generate reforecasts (same interface as other models)"""
        self.to(device)
        self.eval()
        
        forecasts = []
        
        for init_time in init_data.time:
            x0_dict = {var: init_data[var].sel(time=init_time).values for var in self.var_names}
            x0_values = torch.stack([
                torch.tensor(x0_dict[var], dtype=torch.float32, device=device)
                for var in self.var_names
            ], dim=0)
            
            x0_batch = x0_values.unsqueeze(0).expand(ncopy, -1)
            t = torch.linspace(0, n_month, n_month + 1, device=device)
            
            with torch.no_grad():
                if self.noise_mode == 'internal':
                    forecast = self.forward(x0_batch, t, enable_noise=enable_noise)
                else:
                    forecast = self.forward(x0_batch, t, add_noise=add_noise)
                forecasts.append(forecast.detach().cpu().numpy())
        
        forecasts_np = np.stack(forecasts, axis=1)
        
        # Create dataset
        lead_coord = np.arange(n_month + 1)
        init_coord = init_data.time.values
        member_coord = np.arange(ncopy)
        
        data_vars = {}
        for i, var in enumerate(self.var_names):
            data_vars[var] = xr.DataArray(
                forecasts_np[:, :, :, i],
                coords={'lead': lead_coord, 'init_time': init_coord, 'member': member_coord},
                dims=['lead', 'init_time', 'member']
            )
        
        return xr.Dataset(data_vars)
