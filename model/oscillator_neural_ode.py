"""
Oscillator-Constrained Neural ODE

This module implements a Neural ODE with explicit oscillator constraints
to ensure robust oscillatory behavior similar to XRO's recharge oscillator dynamics.

Key features:
1. Explicit harmonic oscillator components
2. Learnable oscillation frequencies and damping
3. Physics-informed loss functions for oscillatory behavior
4. Eigenvalue regularization
"""

import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
import xarray as xr
from typing import List, Optional, Tuple


class HarmonicOscillatorBlock(nn.Module):
    """
    Explicit harmonic oscillator component: d²x/dt² + 2γ*dx/dt + ω²*x = 0
    Implemented as coupled first-order system:
    dx/dt = v
    dv/dt = -2γ*v - ω²*x + forcing
    """
    
    def __init__(self, state_dim: int, n_oscillators: int = 2):
        super().__init__()
        self.state_dim = state_dim
        self.n_oscillators = n_oscillators
        
        # Learnable oscillator parameters
        # Initialize with ENSO-like frequencies (2-5 year periods)
        init_frequencies = torch.tensor([2*np.pi/48, 2*np.pi/36])  # 4-year and 3-year periods (in months)
        self.log_frequencies = nn.Parameter(torch.log(init_frequencies[:n_oscillators]))
        
        # Damping coefficients (small positive values)
        self.log_damping = nn.Parameter(torch.log(torch.tensor([0.1, 0.05])[:n_oscillators]))
        
        # Coupling matrices: how oscillators couple to state variables
        self.position_coupling = nn.Linear(n_oscillators, state_dim, bias=False)
        self.velocity_coupling = nn.Linear(n_oscillators, state_dim, bias=False)
        
        # State to oscillator forcing
        self.state_to_forcing = nn.Linear(state_dim, n_oscillators)
        
        # Initialize oscillator states (position and velocity for each oscillator)
        self.register_buffer('oscillator_state', torch.zeros(2 * n_oscillators))
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute oscillator contribution to dx/dt
        
        Args:
            t: Time tensor [1]
            x: State tensor [batch_size, state_dim]
            
        Returns:
            Oscillator contribution to dx/dt [batch_size, state_dim]
        """
        batch_size = x.shape[0]
        
        # Get oscillator parameters
        frequencies = torch.exp(self.log_frequencies)  # ω
        damping = torch.exp(self.log_damping)          # γ
        
        # Extract oscillator positions and velocities
        # For batch processing, we'll use the same oscillator state for all batch elements
        osc_pos = self.oscillator_state[:self.n_oscillators]  # [n_oscillators]
        osc_vel = self.oscillator_state[self.n_oscillators:]  # [n_oscillators]
        
        # Compute forcing from current state
        forcing = self.state_to_forcing(x)  # [batch_size, n_oscillators]
        
        # Update oscillator dynamics
        # dv/dt = -2γ*v - ω²*x + forcing
        dosc_vel_dt = -2 * damping * osc_vel - frequencies**2 * osc_pos + forcing.mean(0)
        
        # dx/dt = v (for oscillators)
        dosc_pos_dt = osc_vel
        
        # Update oscillator state (in-place for efficiency)
        with torch.no_grad():
            dt = 1.0  # Assume unit time step for state update
            self.oscillator_state[:self.n_oscillators] += dosc_pos_dt * dt
            self.oscillator_state[self.n_oscillators:] += dosc_vel_dt * dt
        
        # Compute contribution to state dynamics
        pos_contribution = self.position_coupling(osc_pos.unsqueeze(0).expand(batch_size, -1))
        vel_contribution = self.velocity_coupling(osc_vel.unsqueeze(0).expand(batch_size, -1))
        
        return pos_contribution + vel_contribution
    
    def get_oscillator_periods(self) -> torch.Tensor:
        """Get current oscillation periods in months"""
        frequencies = torch.exp(self.log_frequencies)
        periods = 2 * np.pi / frequencies
        return periods
    
    def get_oscillator_info(self) -> dict:
        """Get oscillator parameters for analysis"""
        frequencies = torch.exp(self.log_frequencies).detach().cpu().numpy()
        damping = torch.exp(self.log_damping).detach().cpu().numpy()
        periods_months = 2 * np.pi / frequencies
        periods_years = periods_months / 12
        
        return {
            'frequencies': frequencies,
            'damping': damping,
            'periods_months': periods_months,
            'periods_years': periods_years
        }


class SeasonalOscillatorEmbedding(nn.Module):
    """Seasonal embedding that modulates oscillator behavior"""
    
    def __init__(self, embedding_dim: int = 8):
        super().__init__()
        self.embedding_dim = embedding_dim
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Create seasonal embedding from time
        
        Args:
            t: Time tensor [batch_size] or [1]
            
        Returns:
            Seasonal embedding [batch_size, embedding_dim]
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        # Convert time to month (assuming t is in months)
        month = t % 12
        
        # Create Fourier features for seasonal cycle
        features = []
        for k in range(1, self.embedding_dim // 2 + 1):
            features.append(torch.cos(2 * np.pi * k * month / 12))
            features.append(torch.sin(2 * np.pi * k * month / 12))
        
        return torch.stack(features[:self.embedding_dim], dim=-1)


class OscillatorNeuralODEFunc(nn.Module):
    """
    Neural ODE function with explicit oscillator constraints
    
    dx/dt = L(t)*x + N(x) + O(t,x) + ε(t)
    
    Where:
    - L(t): Seasonal linear operator
    - N(x): Nonlinear terms
    - O(t,x): Oscillator dynamics
    - ε(t): Noise (handled externally)
    """
    
    def __init__(self, state_dim: int, var_names: List[str], hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.var_names = var_names
        self.hidden_dim = hidden_dim
        
        # Seasonal embedding
        self.seasonal_embedding = SeasonalOscillatorEmbedding(embedding_dim=8)
        
        # Linear dynamics (seasonally modulated)
        self.linear_net = nn.Sequential(
            nn.Linear(state_dim + 8, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim * state_dim)
        )
        
        # Nonlinear dynamics
        self.nonlinear_net = nn.Sequential(
            nn.Linear(state_dim + 8, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Explicit oscillator components
        self.oscillator_block = HarmonicOscillatorBlock(state_dim, n_oscillators=2)
        
        # Oscillator modulation (seasonal control of oscillator strength)
        self.oscillator_modulation = nn.Sequential(
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 0-1 modulation factor
        )
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute dx/dt with oscillator constraints
        
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
        
        # Linear dynamics (seasonally modulated)
        linear_params = self.linear_net(x_seasonal)  # [batch_size, state_dim^2]
        linear_matrix = linear_params.view(batch_size, self.state_dim, self.state_dim)
        linear_term = torch.bmm(linear_matrix, x.unsqueeze(-1)).squeeze(-1)
        
        # Nonlinear dynamics
        nonlinear_term = self.nonlinear_net(x_seasonal)
        
        # Oscillator dynamics
        oscillator_term = self.oscillator_block(t, x)
        
        # Seasonal modulation of oscillator strength
        osc_modulation = self.oscillator_modulation(seasonal_emb)  # [batch_size, 1]
        oscillator_term = oscillator_term * osc_modulation
        
        # Combine all terms
        dxdt = linear_term + nonlinear_term + oscillator_term
        
        return dxdt
    
    def get_oscillator_info(self) -> dict:
        """Get information about the oscillator components"""
        return self.oscillator_block.get_oscillator_info()


class StochasticOscillatorNeuralODEFunc(OscillatorNeuralODEFunc):
    """
    Oscillator Neural ODE function with noise directly in the vector field (internal noise)
    """
    
    def __init__(self, state_dim: int, var_names: List[str], hidden_dim: int = 64, noise_scale: float = 0.1):
        super().__init__(state_dim, var_names, hidden_dim)
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


class OscillatorNeuralODE(nn.Module):
    """
    Neural ODE with explicit oscillator constraints for ENSO-like dynamics
    Supports both external noise (post-integration) and internal noise (in vector field)
    """
    
    def __init__(self, state_dim: int, var_names: List[str], hidden_dim: int = 64, 
                 noise_mode: str = 'external', noise_scale: float = 0.1):
        super().__init__()
        self.state_dim = state_dim
        self.var_names = var_names
        self.noise_mode = noise_mode  # 'external' or 'internal'
        self.noise_scale = noise_scale
        
        # Create ODE function with or without internal noise
        if noise_mode == 'internal':
            self.ode_func = StochasticOscillatorNeuralODEFunc(state_dim, var_names, hidden_dim, noise_scale)
        else:
            self.ode_func = OscillatorNeuralODEFunc(state_dim, var_names, hidden_dim)
        
        # External noise model for post-integration noise
        if noise_mode == 'external':
            self.noise_net = nn.Sequential(
                nn.Linear(state_dim + 1, 32),  # +1 for seasonal component
                nn.ReLU(),
                nn.Linear(32, state_dim),
                nn.Softplus()  # Ensure positive noise scaling
            )
    
    def forward(self, x0: torch.Tensor, t: torch.Tensor, add_noise: bool = False, enable_noise: bool = False) -> torch.Tensor:
        """
        Forward pass through Neural ODE
        
        Args:
            x0: Initial conditions [batch_size, state_dim]
            t: Time points [n_times]
            add_noise: Whether to add external noise (for external mode)
            enable_noise: Whether to enable internal noise (for internal mode)
            
        Returns:
            Solution trajectory [n_times, batch_size, state_dim]
        """
        # For internal noise mode, noise is already in the ODE function
        # For external noise mode, we add noise after integration
        
        if self.noise_mode == 'internal':
            # Internal noise: noise is part of the vector field
            # The enable_noise parameter controls whether noise is active
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
        Generate simulations using the Neural ODE
        
        Args:
            x0_data: Initial conditions as xarray Dataset
            nyear: Number of years to simulate
            ncopy: Number of ensemble members
            add_noise: Whether to add stochastic noise
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
            time_coord = xr.cftime_range(
                start='2000-01', periods=len(t), freq='MS', calendar='noleap'
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
        Generate reforecasts using the Neural ODE
        
        Args:
            init_data: Initial conditions dataset
            n_month: Forecast lead time in months
            ncopy: Number of ensemble members
            add_noise: Whether to add noise
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
    
    def get_oscillator_info(self) -> dict:
        """Get information about the oscillator components"""
        return self.ode_func.get_oscillator_info()
    
    def compute_eigenvalues_at_state(self, x: torch.Tensor, t: float = 0.0) -> np.ndarray:
        """
        Compute eigenvalues of the Jacobian at a given state
        
        Args:
            x: State tensor [state_dim]
            t: Time value
            
        Returns:
            Eigenvalues as numpy array
        """
        x_tensor = x.clone().detach().requires_grad_(True)
        t_tensor = torch.tensor([t], dtype=torch.float32)
        
        # Compute dx/dt
        dxdt = self.ode_func(t_tensor, x_tensor.unsqueeze(0)).squeeze(0)
        
        # Compute Jacobian
        jacobian = torch.zeros(self.state_dim, self.state_dim)
        
        for i in range(self.state_dim):
            if x_tensor.grad is not None:
                x_tensor.grad.zero_()
            
            dxdt[i].backward(retain_graph=True)
            
            if x_tensor.grad is not None:
                jacobian[i, :] = x_tensor.grad.clone()
        
        # Compute eigenvalues
        from scipy.linalg import eigvals
        eigenvals = eigvals(jacobian.detach().numpy())
        
        return eigenvals


def create_oscillator_loss(model: OscillatorNeuralODE, target_periods: List[float] = [36, 48]) -> callable:
    """
    Create a loss function that encourages oscillatory behavior
    
    Args:
        model: OscillatorNeuralODE model
        target_periods: Target oscillation periods in months
        
    Returns:
        Loss function
    """
    def oscillator_loss(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute oscillator-specific loss terms
        
        Args:
            x: State trajectory [n_times, batch_size, state_dim]
            t: Time points [n_times]
            
        Returns:
            Loss tensor
        """
        device = x.device
        losses = []
        
        # 1. Spectral loss: encourage target frequencies (differentiable)
        # Get current oscillator parameters (these are learnable parameters)
        frequencies = torch.exp(model.ode_func.oscillator_block.log_frequencies)
        current_periods = 2 * np.pi / frequencies  # periods in months
        target_periods_tensor = torch.tensor(target_periods, dtype=torch.float32, device=device)
        
        # Compute distance to target periods (this is differentiable)
        period_loss = torch.mean((current_periods.unsqueeze(1) - target_periods_tensor.unsqueeze(0))**2)
        losses.append(period_loss * 0.01)  # Small weight
        
        # 2. Oscillator amplitude regularization (prevent oscillators from dying)
        osc_state = model.ode_func.oscillator_block.oscillator_state
        # Ensure oscillator state has some activity
        amplitude_loss = torch.exp(-torch.norm(osc_state.detach()) - 1e-8)  # Add small epsilon to prevent NaN
        losses.append(amplitude_loss * 0.001)  # Very small weight
        
        # 3. Trajectory smoothness loss (encourage oscillatory patterns)
        if x.shape[0] > 2:  # Need at least 3 time points
            # Compute second derivatives (acceleration)
            x_diff1 = x[1:] - x[:-1]  # First derivative approximation
            x_diff2 = x_diff1[1:] - x_diff1[:-1]  # Second derivative approximation
            
            # Penalize large accelerations (encourage smooth oscillations)
            smoothness_loss = torch.mean(x_diff2**2) * 0.001
            losses.append(smoothness_loss)
        
        # 4. Oscillatory pattern loss (encourage periodic behavior)
        if x.shape[0] > 12:  # Need enough time points for periodicity
            # Take a subset of the trajectory
            x_subset = x[:12, 0, 0]  # First variable, first batch, first 12 time points
            
            # Compute autocorrelation at different lags
            autocorr_losses = []
            for lag in [3, 6, 9]:  # 3, 6, 9 month lags
                if lag < len(x_subset):
                    x1 = x_subset[:-lag]
                    x2 = x_subset[lag:]
                    
                    # Encourage positive autocorrelation at these lags (oscillatory behavior)
                    if len(x1) > 0:
                        correlation = torch.corrcoef(torch.stack([x1, x2]))[0, 1]
                        # Penalize negative correlations (want some periodicity)
                        if not torch.isnan(correlation):
                            autocorr_loss = torch.clamp(-correlation, 0, 1)  # Only penalize negative correlations
                            autocorr_losses.append(autocorr_loss * 0.001)
            
            if autocorr_losses:
                losses.append(torch.mean(torch.stack(autocorr_losses)))
        
        # 5. Damping regularization (prevent over-damping)
        damping = torch.exp(model.ode_func.oscillator_block.log_damping)
        # Penalize very high damping (want sustained oscillations)
        damping_loss = torch.mean(torch.clamp(damping - 0.5, 0, float('inf'))**2) * 0.001
        losses.append(damping_loss)
        
        # Combine all losses and ensure no NaN
        total_loss = sum(losses)
        
        # Check for NaN and replace with a small positive value
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(0.01, dtype=torch.float32, device=device, requires_grad=True)
        
        return total_loss
    
    return oscillator_loss
