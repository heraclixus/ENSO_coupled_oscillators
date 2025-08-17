"""
Neural ODE model inspired by XRO physical insights
"""

import torch
import torch.nn as nn
import numpy as np
import xarray as xr
from typing import Optional, Dict, Any, List, Tuple
from torchdiffeq import odeint


class SeasonalEmbedding(nn.Module):
    """
    Seasonal embedding layer inspired by XRO's seasonal cycle handling
    """
    def __init__(self, ncycle: int = 12, embed_dim: int = 16):
        super().__init__()
        self.ncycle = ncycle
        self.embed_dim = embed_dim
        
        # Learnable seasonal embeddings
        self.seasonal_embed = nn.Embedding(ncycle, embed_dim)
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time tensor of shape [batch_size] or [batch_size, 1]
        Returns:
            Seasonal embedding of shape [batch_size, embed_dim]
        """
        if t.dim() == 2:
            t = t.squeeze(-1)
        
        # Convert continuous time to seasonal cycle indices
        cycle_idx = ((t * self.ncycle) % self.ncycle).long()
        return self.seasonal_embed(cycle_idx)


class PhysicsInformedBlock(nn.Module):
    """
    Physics-informed neural network block inspired by XRO's structure
    """
    def __init__(self, state_dim: int, hidden_dim: int, seasonal_dim: int = 16):
        super().__init__()
        self.state_dim = state_dim
        
        # Linear dynamics (inspired by XRO's L matrix)
        self.linear_dynamics = nn.Linear(state_dim + seasonal_dim, state_dim)
        
        # Nonlinear dynamics (inspired by XRO's nonlinear terms)
        self.nonlinear_net = nn.Sequential(
            nn.Linear(state_dim + seasonal_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # ENSO-specific nonlinear terms (inspired by XRO's NRO terms)
        # Assumes first two variables are ENSO T and H
        self.enso_nonlinear = nn.Sequential(
            nn.Linear(5 + seasonal_dim, hidden_dim // 2),  # T^2, TH, T^3, T^2H, TH^2
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 2)  # Output for T and H equations
        )
        
    def forward(self, x: torch.Tensor, seasonal_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: State tensor of shape [batch_size, state_dim]
            seasonal_embed: Seasonal embedding of shape [batch_size, seasonal_dim]
        Returns:
            Time derivative dx/dt of shape [batch_size, state_dim]
        """
        batch_size = x.shape[0]
        
        # Combine state and seasonal information
        x_seasonal = torch.cat([x, seasonal_embed], dim=-1)
        
        # Linear dynamics
        linear_term = self.linear_dynamics(x_seasonal)
        
        # General nonlinear dynamics
        nonlinear_term = self.nonlinear_net(x_seasonal)
        
        # ENSO-specific nonlinear terms (if we have at least 2 variables)
        enso_term = torch.zeros_like(x)
        if self.state_dim >= 2:
            T, H = x[:, 0:1], x[:, 1:2]  # ENSO T and H
            
            # Create ENSO nonlinear features: T^2, TH, T^3, T^2H, TH^2
            T2 = T * T
            TH = T * H
            T3 = T * T * T
            T2H = T * T * H
            TH2 = T * H * H
            
            enso_features = torch.cat([T2, TH, T3, T2H, TH2], dim=-1)
            enso_features_seasonal = torch.cat([enso_features, seasonal_embed], dim=-1)
            
            enso_nonlinear_out = self.enso_nonlinear(enso_features_seasonal)
            enso_term[:, :2] = enso_nonlinear_out
        
        return linear_term + nonlinear_term + enso_term


class ODEFunc(nn.Module):
    """
    ODE function for neural ODE integration
    """
    def __init__(self, state_dim: int, hidden_dim: int = 64, seasonal_dim: int = 16, ncycle: int = 12):
        super().__init__()
        self.state_dim = state_dim
        self.seasonal_embedding = SeasonalEmbedding(ncycle, seasonal_dim)
        self.physics_block = PhysicsInformedBlock(state_dim, hidden_dim, seasonal_dim)
        
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time tensor of shape [1] (for odeint)
            x: State tensor of shape [batch_size, state_dim]
        Returns:
            Time derivative dx/dt of shape [batch_size, state_dim]
        """
        # Get seasonal embedding for current time
        t_batch = t.expand(x.shape[0])
        seasonal_embed = self.seasonal_embedding(t_batch)
        
        # Compute dynamics
        return self.physics_block(x, seasonal_embed)


class NeuralODE(nn.Module):
    """
    Neural ODE model for climate dynamics inspired by XRO
    """
    def __init__(self, 
                 state_dim: int,
                 hidden_dim: int = 64,
                 seasonal_dim: int = 16,
                 ncycle: int = 12,
                 var_names: Optional[List[str]] = None):
        super().__init__()
        
        self.state_dim = state_dim
        self.ncycle = ncycle
        self.var_names = var_names or [f'X{i+1}' for i in range(state_dim)]
        
        # ODE function
        self.ode_func = ODEFunc(state_dim, hidden_dim, seasonal_dim, ncycle)
        
        # Noise model (inspired by XRO's stochastic component)
        self.noise_net = nn.Sequential(
            nn.Linear(state_dim + seasonal_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, state_dim),
            nn.Softplus()  # Ensure positive noise
        )
        
    def forward(self, x0: torch.Tensor, t: torch.Tensor, 
                add_noise: bool = False, noise_scale: float = 1.0) -> torch.Tensor:
        """
        Forward integration of the neural ODE
        
        Args:
            x0: Initial conditions [batch_size, state_dim]
            t: Time points [n_times]
            add_noise: Whether to add stochastic noise
            noise_scale: Scale of the noise
        Returns:
            Solution trajectory [batch_size, n_times, state_dim]
        """
        # Integrate ODE
        solution = odeint(self.ode_func, x0, t, method='dopri5')
        solution = solution.permute(1, 0, 2)  # [batch_size, n_times, state_dim]
        
        # Add noise if requested
        if add_noise:
            noise = self._generate_noise(solution, t, noise_scale)
            solution = solution + noise
            
        return solution
    
    def _generate_noise(self, solution: torch.Tensor, t: torch.Tensor, 
                       noise_scale: float) -> torch.Tensor:
        """Generate state-dependent noise"""
        batch_size, n_times, state_dim = solution.shape
        noise = torch.zeros_like(solution)
        
        for i, time_point in enumerate(t):
            # Get seasonal embedding
            t_batch = time_point.expand(batch_size)
            seasonal_embed = self.ode_func.seasonal_embedding(t_batch)
            
            # Combine state and seasonal info
            state_seasonal = torch.cat([solution[:, i], seasonal_embed], dim=-1)
            
            # Generate noise standard deviation
            noise_std = self.noise_net(state_seasonal)
            
            # Sample noise
            noise[:, i] = torch.randn_like(solution[:, i]) * noise_std * noise_scale
            
        return noise
    
    def simulate(self, x0_data: xr.Dataset, nyear: int = 10, ncopy: int = 1, 
                 add_noise: bool = True, noise_scale: float = 1.0,
                 device: str = 'cpu') -> xr.Dataset:
        """
        Simulate the model (similar to XRO.simulate)
        
        Args:
            x0_data: Initial conditions as xarray Dataset
            nyear: Number of years to simulate
            ncopy: Number of ensemble members
            add_noise: Whether to add stochastic noise
            noise_scale: Scale of the noise
            device: Device to run on
        Returns:
            Simulated trajectories as xarray Dataset
        """
        self.eval()
        self.to(device)
        
        # Convert initial conditions to tensor
        x0_np = np.stack([x0_data[var].values for var in self.var_names], axis=0)
        x0 = torch.tensor(x0_np, dtype=torch.float32, device=device)
        x0 = x0.unsqueeze(0).repeat(ncopy, 1)  # [ncopy, state_dim]
        
        # Create time points (monthly)
        n_times = nyear * self.ncycle
        t = torch.linspace(0, nyear, n_times + 1, device=device)
        
        with torch.no_grad():
            # Simulate
            solution = self.forward(x0, t, add_noise=add_noise, noise_scale=noise_scale)
            
            # Convert back to numpy
            solution_np = solution.cpu().numpy()  # [ncopy, n_times+1, state_dim]
        
        # Create time coordinates
        if self.ncycle == 12:
            time_coords = xr.cftime_range('0001-01', periods=n_times+1, freq='MS')
        else:
            time_coords = np.linspace(0, nyear, n_times+1)
        
        # Create xarray Dataset
        coords = {
            'time': time_coords,
            'member': np.arange(ncopy)
        }
        
        data_vars = {}
        for i, var_name in enumerate(self.var_names):
            data_vars[var_name] = (['member', 'time'], solution_np[:, :, i])
        
        return xr.Dataset(data_vars, coords=coords)
    
    def reforecast(self, init_data: xr.Dataset, n_month: int = 12, ncopy: int = 1,
                   add_noise: bool = True, noise_scale: float = 1.0,
                   device: str = 'cpu') -> xr.Dataset:
        """
        Generate reforecasts (similar to XRO.reforecast)
        
        Args:
            init_data: Initial conditions dataset with time dimension
            n_month: Number of months to forecast
            ncopy: Number of ensemble members
            add_noise: Whether to add stochastic noise
            noise_scale: Scale of the noise
            device: Device to run on
        Returns:
            Forecast dataset with init and lead dimensions
        """
        self.eval()
        self.to(device)
        
        n_inits = len(init_data.time)
        n_leads = n_month + 1
        
        # Initialize output arrays
        forecast_data = {}
        for var_name in self.var_names:
            forecast_data[var_name] = np.zeros((n_inits, n_leads, ncopy))
        
        with torch.no_grad():
            for i, init_time in enumerate(init_data.time):
                # Get initial condition for this time
                x0_dict = {var: init_data[var].sel(time=init_time).values 
                          for var in self.var_names}
                x0_np = np.stack([x0_dict[var] for var in self.var_names], axis=0)
                x0 = torch.tensor(x0_np, dtype=torch.float32, device=device)
                x0 = x0.unsqueeze(0).repeat(ncopy, 1)  # [ncopy, state_dim]
                
                # Create forecast time points
                t = torch.linspace(0, n_month/12, n_leads, device=device)
                
                # Generate forecast
                solution = self.forward(x0, t, add_noise=add_noise, noise_scale=noise_scale)
                solution_np = solution.cpu().numpy()  # [ncopy, n_leads, state_dim]
                
                # Store results
                for j, var_name in enumerate(self.var_names):
                    forecast_data[var_name][i, :, :] = solution_np[:, :, j].T
        
        # Create coordinates
        coords = {
            'init': init_data.time,
            'lead': np.arange(n_leads),
            'member': np.arange(ncopy)
        }
        
        # Create data variables
        data_vars = {}
        for var_name in self.var_names:
            if ncopy == 1:
                data_vars[var_name] = (['init', 'lead'], forecast_data[var_name][:, :, 0])
            else:
                data_vars[var_name] = (['init', 'lead', 'member'], forecast_data[var_name])
        
        result = xr.Dataset(data_vars, coords=coords)
        
        if ncopy == 1:
            result = result.squeeze('member', drop=True)
            
        return result
