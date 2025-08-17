"""
Stochastic Neural ODE with noise integrated into the vector field
"""

import torch
import torch.nn as nn
import numpy as np
import xarray as xr
from typing import Optional, Dict, Any, List, Tuple
from torchdiffeq import odeint


class SeasonalNoiseEmbedding(nn.Module):
    """
    Seasonal noise embedding that generates time-dependent stochastic terms
    """
    def __init__(self, state_dim: int, ncycle: int = 12, noise_dim: int = 16):
        super().__init__()
        self.state_dim = state_dim
        self.ncycle = ncycle
        self.noise_dim = noise_dim
        
        # Seasonal noise amplitude (learnable)
        self.seasonal_noise_amp = nn.Parameter(torch.ones(state_dim, ncycle) * 0.1)
        
        # Noise embedding network
        self.noise_embed_net = nn.Sequential(
            nn.Linear(1, noise_dim),  # Time input
            nn.Tanh(),
            nn.Linear(noise_dim, noise_dim),
            nn.Tanh(),
            nn.Linear(noise_dim, state_dim),
            nn.Tanh()
        )
        
    def forward(self, t: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Generate seasonal noise vector
        
        Args:
            t: Time tensor [1] (current time)
            batch_size: Number of samples in batch
        Returns:
            Noise vector [batch_size, state_dim]
        """
        # Get seasonal cycle index
        cycle_phase = (t * self.ncycle) % self.ncycle
        cycle_idx = cycle_phase.long()
        
        # Get seasonal amplitude
        seasonal_amp = self.seasonal_noise_amp[:, cycle_idx].squeeze()  # [state_dim]
        
        # Generate time-dependent noise pattern
        t_input = t.unsqueeze(0).unsqueeze(-1)  # [1, 1]
        noise_pattern = self.noise_embed_net(t_input).squeeze(0)  # [state_dim]
        
        # Combine seasonal amplitude with learned pattern
        noise_base = seasonal_amp * noise_pattern  # [state_dim]
        
        # Sample random noise and scale
        random_noise = torch.randn(batch_size, self.state_dim, device=t.device)
        
        # Apply learned noise pattern
        noise_vector = random_noise * noise_base.unsqueeze(0)  # [batch_size, state_dim]
        
        return noise_vector


class StochasticODEFunc(nn.Module):
    """
    ODE function with integrated stochastic terms
    dx/dt = f(x, t) + σ(t) * ε(t)
    """
    def __init__(self, state_dim: int, hidden_dim: int = 64, ncycle: int = 12, 
                 noise_scale: float = 1.0):
        super().__init__()
        self.state_dim = state_dim
        self.noise_scale = noise_scale
        
        # Deterministic dynamics (similar to physics-informed structure)
        self.linear_net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Nonlinear dynamics
        self.nonlinear_net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # ENSO-specific terms (if state_dim >= 2)
        if state_dim >= 2:
            self.enso_net = nn.Sequential(
                nn.Linear(5 + 1, hidden_dim // 2),  # T^2, TH, T^3, T^2H, TH^2 + time
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 2)  # For T and H equations
            )
        
        # Seasonal noise generator
        self.noise_generator = SeasonalNoiseEmbedding(state_dim, ncycle)
        
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute dx/dt = f(x, t) + σ(t) * ε(t)
        
        Args:
            t: Time tensor [1]
            x: State tensor [batch_size, state_dim]
        Returns:
            Time derivative [batch_size, state_dim]
        """
        batch_size = x.shape[0]
        
        # Expand time for batch
        t_expanded = t.expand(batch_size, 1)  # [batch_size, 1]
        
        # Combine state and time
        x_t = torch.cat([x, t_expanded], dim=-1)  # [batch_size, state_dim + 1]
        
        # Deterministic terms
        linear_term = self.linear_net(x_t)
        nonlinear_term = self.nonlinear_net(x_t)
        
        # ENSO-specific terms
        enso_term = torch.zeros_like(x)
        if self.state_dim >= 2:
            T, H = x[:, 0:1], x[:, 1:2]
            
            # ENSO nonlinear features
            T2 = T * T
            TH = T * H
            T3 = T * T * T
            T2H = T * T * H
            TH2 = T * H * H
            
            enso_features = torch.cat([T2, TH, T3, T2H, TH2, t_expanded], dim=-1)
            enso_out = self.enso_net(enso_features)
            enso_term[:, :2] = enso_out
        
        # Deterministic dynamics
        deterministic = linear_term + nonlinear_term + enso_term
        
        # Stochastic term - noise integrated into vector field
        stochastic = self.noise_generator(t, batch_size) * self.noise_scale
        
        return deterministic + stochastic


class StochasticNeuralODE(nn.Module):
    """
    Neural ODE with noise integrated directly into the vector field
    """
    def __init__(self, 
                 state_dim: int,
                 hidden_dim: int = 64,
                 ncycle: int = 12,
                 noise_scale: float = 1.0,
                 var_names: Optional[List[str]] = None):
        super().__init__()
        
        self.state_dim = state_dim
        self.ncycle = ncycle
        self.noise_scale = noise_scale
        self.var_names = var_names or [f'X{i+1}' for i in range(state_dim)]
        
        # Stochastic ODE function
        self.ode_func = StochasticODEFunc(state_dim, hidden_dim, ncycle, noise_scale)
        
    def forward(self, x0: torch.Tensor, t: torch.Tensor, 
                enable_noise: bool = True) -> torch.Tensor:
        """
        Forward integration with stochastic vector field
        
        Args:
            x0: Initial conditions [batch_size, state_dim]
            t: Time points [n_times]
            enable_noise: Whether to include stochastic terms
        Returns:
            Solution trajectory [batch_size, n_times, state_dim]
        """
        # Temporarily disable noise if requested
        original_noise_scale = self.ode_func.noise_scale
        if not enable_noise:
            self.ode_func.noise_scale = 0.0
        
        try:
            # Integrate stochastic ODE
            solution = odeint(self.ode_func, x0, t, method='euler', options={'step_size': 0.01})
            solution = solution.permute(1, 0, 2)  # [batch_size, n_times, state_dim]
        finally:
            # Restore original noise scale
            self.ode_func.noise_scale = original_noise_scale
            
        return solution
    
    def set_noise_scale(self, noise_scale: float):
        """Set the noise scale for the stochastic terms"""
        self.noise_scale = noise_scale
        self.ode_func.noise_scale = noise_scale
    
    def simulate(self, x0_data: xr.Dataset, nyear: int = 10, ncopy: int = 1, 
                 enable_noise: bool = True, noise_scale: float = None,
                 device: str = 'cpu') -> xr.Dataset:
        """
        Simulate the stochastic model (similar to XRO.simulate)
        
        Args:
            x0_data: Initial conditions as xarray Dataset
            nyear: Number of years to simulate
            ncopy: Number of ensemble members
            enable_noise: Whether to include stochastic forcing
            noise_scale: Override noise scale (if None, use model's default)
            device: Device to run on
        Returns:
            Simulated trajectories as xarray Dataset
        """
        self.eval()
        self.to(device)
        
        # Set noise scale if provided
        if noise_scale is not None:
            original_scale = self.noise_scale
            self.set_noise_scale(noise_scale)
        
        try:
            # Convert initial conditions to tensor
            x0_np = np.stack([x0_data[var].values for var in self.var_names], axis=0)
            x0 = torch.tensor(x0_np, dtype=torch.float32, device=device)
            x0 = x0.unsqueeze(0).repeat(ncopy, 1)  # [ncopy, state_dim]
            
            # Create time points (monthly)
            n_times = nyear * self.ncycle
            t = torch.linspace(0, nyear, n_times + 1, device=device)
            
            with torch.no_grad():
                # Simulate with different random seeds for each ensemble member
                all_solutions = []
                for i in range(ncopy):
                    # Set different random seed for each member
                    torch.manual_seed(i + 1000)  # Offset to avoid seed=0
                    
                    solution = self.forward(x0[i:i+1], t, enable_noise=enable_noise)
                    all_solutions.append(solution)
                
                # Combine all ensemble members
                solution_combined = torch.cat(all_solutions, dim=0)  # [ncopy, n_times+1, state_dim]
                solution_np = solution_combined.cpu().numpy()
            
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
            
        finally:
            # Restore original noise scale if it was changed
            if noise_scale is not None:
                self.set_noise_scale(original_scale)
    
    def reforecast(self, init_data: xr.Dataset, n_month: int = 12, ncopy: int = 1,
                   enable_noise: bool = True, noise_scale: float = None,
                   device: str = 'cpu') -> xr.Dataset:
        """
        Generate reforecasts with stochastic vector field
        
        Args:
            init_data: Initial conditions dataset with time dimension
            n_month: Number of months to forecast
            ncopy: Number of ensemble members
            enable_noise: Whether to include stochastic forcing
            noise_scale: Override noise scale
            device: Device to run on
        Returns:
            Forecast dataset with init and lead dimensions
        """
        self.eval()
        self.to(device)
        
        # Set noise scale if provided
        if noise_scale is not None:
            original_scale = self.noise_scale
            self.set_noise_scale(noise_scale)
        
        try:
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
                    
                    # Create forecast time points
                    t = torch.linspace(0, n_month/12, n_leads, device=device)
                    
                    # Generate ensemble forecasts
                    for j in range(ncopy):
                        # Different random seed for each ensemble member
                        torch.manual_seed(i * 1000 + j + 2000)
                        
                        solution = self.forward(x0.unsqueeze(0), t, enable_noise=enable_noise)
                        solution_np = solution.cpu().numpy()  # [1, n_leads, state_dim]
                        
                        # Store results
                        for k, var_name in enumerate(self.var_names):
                            forecast_data[var_name][i, :, j] = solution_np[0, :, k]
            
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
            
        finally:
            # Restore original noise scale if it was changed
            if noise_scale is not None:
                self.set_noise_scale(original_scale)
    
    def get_noise_characteristics(self, t_points: torch.Tensor, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """
        Analyze the learned noise characteristics over time
        
        Args:
            t_points: Time points to analyze
            device: Device to run on
        Returns:
            Dictionary with noise statistics
        """
        self.eval()
        self.to(device)
        
        noise_samples = []
        seasonal_amps = []
        
        with torch.no_grad():
            for t in t_points:
                # Generate noise samples
                noise = self.ode_func.noise_generator(t, 100)  # 100 samples
                noise_samples.append(noise)
                
                # Get seasonal amplitude
                cycle_phase = (t * self.ncycle) % self.ncycle
                cycle_idx = cycle_phase.long()
                seasonal_amp = self.ode_func.noise_generator.seasonal_noise_amp[:, cycle_idx]
                seasonal_amps.append(seasonal_amp)
        
        noise_tensor = torch.stack(noise_samples, dim=0)  # [n_times, 100, state_dim]
        seasonal_tensor = torch.stack(seasonal_amps, dim=0)  # [n_times, state_dim]
        
        return {
            'noise_std': noise_tensor.std(dim=1),  # [n_times, state_dim]
            'noise_mean': noise_tensor.mean(dim=1),  # [n_times, state_dim]
            'seasonal_amplitude': seasonal_tensor,  # [n_times, state_dim]
            'time_points': t_points
        }
