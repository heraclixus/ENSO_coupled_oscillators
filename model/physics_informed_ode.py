"""
Physics-informed Neural ODE that closely follows XRO's mathematical structure
"""

import torch
import torch.nn as nn
import numpy as np
import xarray as xr
from typing import Optional, Dict, Any, List, Tuple
from torchdiffeq import odeint


class SeasonalLinearOperator(nn.Module):
    """
    Learnable seasonal linear operator inspired by XRO's L(t) matrix
    """
    def __init__(self, state_dim: int, ncycle: int = 12, ac_order: int = 2):
        super().__init__()
        self.state_dim = state_dim
        self.ncycle = ncycle
        self.ac_order = ac_order
        
        # Fourier coefficients for seasonal modulation
        # L = L_0 + L_1^c cos(ωt) + L_1^s sin(ωt) + L_2^c cos(2ωt) + L_2^s sin(2ωt) + ...
        n_coeffs = 2 * ac_order + 1  # L_0, L_1^c, L_1^s, L_2^c, L_2^s, ...
        self.fourier_coeffs = nn.Parameter(torch.randn(state_dim, state_dim, n_coeffs) * 0.1)
        
        self.omega = 2 * np.pi  # Annual frequency
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute seasonal linear operator L(t)
        
        Args:
            t: Time tensor [batch_size] or scalar
        Returns:
            Linear operator L(t) of shape [batch_size, state_dim, state_dim]
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        batch_size = t.shape[0]
        
        # Start with L_0 (annual mean)
        L = self.fourier_coeffs[:, :, 0].unsqueeze(0).expand(batch_size, -1, -1).clone()
        
        # Add seasonal components
        for k in range(1, self.ac_order + 1):
            cos_term = torch.cos(k * self.omega * t).unsqueeze(-1).unsqueeze(-1)
            sin_term = torch.sin(k * self.omega * t).unsqueeze(-1).unsqueeze(-1)
            
            L = L + cos_term * self.fourier_coeffs[:, :, k].unsqueeze(0)
            L = L + sin_term * self.fourier_coeffs[:, :, k + self.ac_order].unsqueeze(0)
        
        return L


class NonlinearTerms(nn.Module):
    """
    Nonlinear terms inspired by XRO's structure
    """
    def __init__(self, state_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.state_dim = state_dim
        
        # Quadratic terms (X^2)
        self.quadratic_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Cubic terms (X^3)  
        self.cubic_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # ENSO-specific nonlinear terms (for first two variables)
        if state_dim >= 2:
            self.enso_T_net = nn.Sequential(
                nn.Linear(5, hidden_dim // 2),  # T^2, TH, T^3, T^2H, TH^2
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )
            
            self.enso_H_net = nn.Sequential(
                nn.Linear(5, hidden_dim // 2),  # T^2, TH, T^3, T^2H, TH^2
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute nonlinear terms
        
        Args:
            x: State tensor [batch_size, state_dim]
        Returns:
            Nonlinear terms [batch_size, state_dim]
        """
        batch_size = x.shape[0]
        
        # General quadratic and cubic terms
        quad_input = x * x  # Element-wise square
        cubic_input = x * x * x  # Element-wise cube
        
        quad_terms = self.quadratic_net(quad_input)
        cubic_terms = self.cubic_net(cubic_input)
        
        nonlinear = quad_terms + cubic_terms
        
        # ENSO-specific terms (if we have at least 2 variables)
        if self.state_dim >= 2:
            T, H = x[:, 0:1], x[:, 1:2]  # ENSO T and H
            
            # Create ENSO nonlinear features: T^2, TH, T^3, T^2H, TH^2
            T2 = T * T
            TH = T * H
            T3 = T * T * T
            T2H = T * T * H
            TH2 = T * H * H
            
            enso_features = torch.cat([T2, TH, T3, T2H, TH2], dim=-1)
            
            # Apply ENSO-specific networks
            enso_T_term = self.enso_T_net(enso_features)
            enso_H_term = self.enso_H_net(enso_features)
            
            # Add to the first two components
            nonlinear[:, 0:1] += enso_T_term
            nonlinear[:, 1:2] += enso_H_term
        
        return nonlinear


class NoiseModel(nn.Module):
    """
    State and seasonally dependent noise model
    """
    def __init__(self, state_dim: int, ncycle: int = 12):
        super().__init__()
        self.state_dim = state_dim
        self.ncycle = ncycle
        
        # Seasonal noise amplitude (inspired by XRO's xi_stdac)
        self.seasonal_noise_amp = nn.Parameter(torch.ones(state_dim, ncycle) * 0.1)
        
        # State-dependent noise scaling
        self.state_noise_net = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.Softplus()
        )
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Generate noise standard deviation
        
        Args:
            x: State tensor [batch_size, state_dim]
            t: Time tensor [batch_size]
        Returns:
            Noise std [batch_size, state_dim]
        """
        batch_size = x.shape[0]
        
        # Get seasonal cycle index
        cycle_idx = ((t * self.ncycle) % self.ncycle).long()
        
        # Get seasonal noise amplitude
        seasonal_amp = self.seasonal_noise_amp[:, cycle_idx].T  # [batch_size, state_dim]
        
        # Get state-dependent scaling
        state_scaling = self.state_noise_net(torch.abs(x))
        
        return seasonal_amp * state_scaling


class PhysicsInformedODEFunc(nn.Module):
    """
    Physics-informed ODE function following XRO's structure:
    dx/dt = L(t) * x + N(x) + ξ(t)
    """
    def __init__(self, state_dim: int, ncycle: int = 12, ac_order: int = 2, hidden_dim: int = 32):
        super().__init__()
        
        self.linear_operator = SeasonalLinearOperator(state_dim, ncycle, ac_order)
        self.nonlinear_terms = NonlinearTerms(state_dim, hidden_dim)
        self.noise_model = NoiseModel(state_dim, ncycle)
        
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute dx/dt = L(t) * x + N(x)
        
        Args:
            t: Time tensor [1] (for odeint)
            x: State tensor [batch_size, state_dim]
        Returns:
            Time derivative [batch_size, state_dim]
        """
        batch_size = x.shape[0]
        t_batch = t.expand(batch_size)
        
        # Linear term: L(t) * x
        L_t = self.linear_operator(t_batch)  # [batch_size, state_dim, state_dim]
        linear_term = torch.bmm(L_t, x.unsqueeze(-1)).squeeze(-1)  # [batch_size, state_dim]
        
        # Nonlinear terms: N(x)
        nonlinear_term = self.nonlinear_terms(x)
        
        return linear_term + nonlinear_term


class PhysicsInformedODE(nn.Module):
    """
    Physics-informed Neural ODE that closely follows XRO's mathematical structure
    """
    def __init__(self, 
                 state_dim: int,
                 ncycle: int = 12,
                 ac_order: int = 2,
                 hidden_dim: int = 32,
                 var_names: Optional[List[str]] = None):
        super().__init__()
        
        self.state_dim = state_dim
        self.ncycle = ncycle
        self.ac_order = ac_order
        self.var_names = var_names or [f'X{i+1}' for i in range(state_dim)]
        
        # ODE function
        self.ode_func = PhysicsInformedODEFunc(state_dim, ncycle, ac_order, hidden_dim)
        
    def forward(self, x0: torch.Tensor, t: torch.Tensor, 
                add_noise: bool = False, noise_scale: float = 1.0) -> torch.Tensor:
        """
        Forward integration of the physics-informed ODE
        
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
        """Generate state and seasonally dependent noise"""
        batch_size, n_times, state_dim = solution.shape
        noise = torch.zeros_like(solution)
        
        for i, time_point in enumerate(t):
            t_batch = time_point.expand(batch_size)
            
            # Get noise standard deviation
            noise_std = self.ode_func.noise_model(solution[:, i], t_batch)
            
            # Sample noise
            noise[:, i] = torch.randn_like(solution[:, i]) * noise_std * noise_scale
            
        return noise
    
    def get_linear_operator(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get the linear operator L(t) at specific times
        
        Args:
            t: Time points [n_times]
        Returns:
            Linear operators [n_times, state_dim, state_dim]
        """
        return self.ode_func.linear_operator(t)
    
    def get_seasonal_components(self) -> Dict[str, torch.Tensor]:
        """
        Extract seasonal components of the linear operator
        
        Returns:
            Dictionary with Fourier coefficients
        """
        coeffs = self.ode_func.linear_operator.fourier_coeffs
        
        components = {
            'L0': coeffs[:, :, 0],  # Annual mean
        }
        
        for k in range(1, self.ac_order + 1):
            components[f'L{k}_cos'] = coeffs[:, :, k]
            components[f'L{k}_sin'] = coeffs[:, :, k + self.ac_order]
        
        return components
    
    def simulate(self, x0_data: xr.Dataset, nyear: int = 10, ncopy: int = 1, 
                 add_noise: bool = True, noise_scale: float = 1.0,
                 device: str = 'cpu') -> xr.Dataset:
        """
        Simulate the model (similar to XRO.simulate)
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
