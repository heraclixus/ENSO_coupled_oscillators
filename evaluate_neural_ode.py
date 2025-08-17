"""
Evaluation script for Neural ODE models - comparison with XRO
"""

import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import argparse
import os

from model import NeuralODE, PhysicsInformedODE, StochasticNeuralODE
from xro.XRO import XRO
from utils import calc_forecast_skill


def load_trained_model(checkpoint_path: str, model_type: str, var_names: List[str], device: str = 'auto') -> torch.nn.Module:
    """Load a trained model from checkpoint"""
    # Auto-detect device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    state_dim = len(var_names)
    if model_type == 'neural_ode':
        model = NeuralODE(state_dim=state_dim, var_names=var_names)
    elif model_type == 'stochastic':
        model = StochasticNeuralODE(state_dim=state_dim, var_names=var_names)
    else:
        model = PhysicsInformedODE(state_dim=state_dim, var_names=var_names)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def evaluate_forecast_skill(model: torch.nn.Module, data: xr.Dataset, 
                           n_month: int = 21, device: str = 'cpu') -> xr.Dataset:
    """
    Evaluate forecast skill of neural ODE model
    
    Args:
        model: Trained neural ODE model
        data: Evaluation dataset
        n_month: Forecast horizon in months
        device: Device to run on
    Returns:
        Forecast dataset
    """
    model.to(device)
    model.eval()
    
    # Generate forecasts
    if hasattr(model, 'reforecast') and 'enable_noise' in model.reforecast.__code__.co_varnames:
        # Stochastic model
        forecasts = model.reforecast(
            init_data=data,
            n_month=n_month,
            ncopy=1,
            enable_noise=False,  # Deterministic for evaluation
            device=device
        )
    else:
        # Regular models
        forecasts = model.reforecast(
            init_data=data,
            n_month=n_month,
            ncopy=1,
            add_noise=False,
            device=device
        )
    
    return forecasts


def compare_with_xro(neural_ode_forecasts: xr.Dataset, obs_data: xr.Dataset,
                    xro_model: XRO, train_data: xr.Dataset) -> Dict:
    """
    Compare Neural ODE forecasts with XRO model
    
    Args:
        neural_ode_forecasts: Neural ODE forecast dataset
        obs_data: Observational data for verification
        xro_model: Fitted XRO model
        train_data: Training data for XRO
    Returns:
        Comparison results dictionary
    """
    # Fit XRO model
    xro_fit = xro_model.fit_matrix(train_data, maskb=['IOD'], maskNT=['T2', 'TH'])
    
    # Generate XRO forecasts
    xro_forecasts = xro_model.reforecast(
        fit_ds=xro_fit,
        init_ds=obs_data,
        n_month=neural_ode_forecasts.lead.max().item(),
        ncopy=1,
        noise_type='zero'
    )
    
    # Calculate forecast skills
    neural_ode_skill = calc_forecast_skill(
        neural_ode_forecasts, obs_data, 
        metric='acc', is_mv3=True, by_month=False,
        verify_periods=slice('1979-01', '2022-12')
    )
    
    xro_skill = calc_forecast_skill(
        xro_forecasts, obs_data,
        metric='acc', is_mv3=True, by_month=False,
        verify_periods=slice('1979-01', '2022-12')
    )
    
    # Calculate RMSE
    neural_ode_rmse = calc_forecast_skill(
        neural_ode_forecasts, obs_data,
        metric='rmse', is_mv3=True, by_month=False,
        verify_periods=slice('1979-01', '2022-12')
    )
    
    xro_rmse = calc_forecast_skill(
        xro_forecasts, obs_data,
        metric='rmse', is_mv3=True, by_month=False,
        verify_periods=slice('1979-01', '2022-12')
    )
    
    return {
        'neural_ode_skill': neural_ode_skill,
        'xro_skill': xro_skill,
        'neural_ode_rmse': neural_ode_rmse,
        'xro_rmse': xro_rmse,
        'neural_ode_forecasts': neural_ode_forecasts,
        'xro_forecasts': xro_forecasts
    }


def plot_forecast_comparison(results: Dict, var_name: str = 'Nino34', 
                           save_path: str = None):
    """Plot forecast skill comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Correlation skill
    ax = axes[0]
    results['neural_ode_skill'][var_name].plot(ax=ax, label='Neural ODE', color='red', linewidth=2)
    results['xro_skill'][var_name].plot(ax=ax, label='XRO', color='blue', linewidth=2)
    
    ax.set_ylabel('Correlation Skill')
    ax.set_xlabel('Forecast Lead (months)')
    ax.set_title(f'{var_name} Forecast Correlation Skill')
    ax.axhline(0.5, ls='--', color='black', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # RMSE
    ax = axes[1]
    results['neural_ode_rmse'][var_name].plot(ax=ax, label='Neural ODE', color='red', linewidth=2)
    results['xro_rmse'][var_name].plot(ax=ax, label='XRO', color='blue', linewidth=2)
    
    ax.set_ylabel('RMSE (°C)')
    ax.set_xlabel('Forecast Lead (months)')
    ax.set_title(f'{var_name} Forecast RMSE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # plt.show()  # Commented out to avoid popup during training


def plot_forecast_plume(results: Dict, init_dates: List[str], var_name: str = 'Nino34',
                       obs_data: xr.Dataset = None, save_path: str = None):
    """Plot forecast plume comparison"""
    n_dates = len(init_dates)
    fig, axes = plt.subplots(n_dates, 1, figsize=(12, 4*n_dates))
    
    if n_dates == 1:
        axes = [axes]
    
    for i, init_date in enumerate(init_dates):
        ax = axes[i]
        
        # Neural ODE forecast
        neural_ode_fcst = results['neural_ode_forecasts'][var_name].sel(init=init_date)
        xro_fcst = results['xro_forecasts'][var_name].sel(init=init_date)
        
        # Create time axis for plotting
        lead_months = neural_ode_fcst.lead.values
        init_time = neural_ode_fcst.init.values
        
        # Plot forecasts
        ax.plot(lead_months, neural_ode_fcst.values, 'r-', linewidth=2, label='Neural ODE')
        ax.plot(lead_months, xro_fcst.values, 'b-', linewidth=2, label='XRO')
        
        # Plot observations if available
        if obs_data is not None:
            # Get observation window
            obs_window = obs_data[var_name].sel(
                time=slice(init_date, None)
            ).isel(time=slice(0, len(lead_months)))
            
            if len(obs_window) > 0:
                ax.plot(lead_months[:len(obs_window)], obs_window.values, 
                       'k-', linewidth=2, alpha=0.7, label='Observation')
        
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.3)
        ax.axhline(-0.5, color='blue', linestyle='--', alpha=0.3)
        
        ax.set_xlabel('Lead (months)')
        ax.set_ylabel(f'{var_name} (°C)')
        ax.set_title(f'Forecast initialized from {init_date}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # plt.show()  # Commented out to avoid popup during training


def analyze_model_components(model: torch.nn.Module, save_path: str = None):
    """Analyze learned model components (for PhysicsInformedODE)"""
    if not isinstance(model, PhysicsInformedODE):
        print("Component analysis only available for PhysicsInformedODE")
        return
    
    # Get seasonal components
    components = model.get_seasonal_components()
    
    # Plot seasonal linear operator components
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Annual mean (L0)
    im1 = axes[0].imshow(components['L0'].detach().numpy(), cmap='RdBu_r', aspect='auto')
    axes[0].set_title('Annual Mean Linear Operator (L₀)')
    axes[0].set_xlabel('State Variable')
    axes[0].set_ylabel('State Variable')
    plt.colorbar(im1, ax=axes[0])
    
    # Annual cycle cosine (L1_cos)
    if 'L1_cos' in components:
        im2 = axes[1].imshow(components['L1_cos'].detach().numpy(), cmap='RdBu_r', aspect='auto')
        axes[1].set_title('Annual Cycle Cosine (L₁ᶜ)')
        axes[1].set_xlabel('State Variable')
        axes[1].set_ylabel('State Variable')
        plt.colorbar(im2, ax=axes[1])
    
    # Annual cycle sine (L1_sin)
    if 'L1_sin' in components:
        im3 = axes[2].imshow(components['L1_sin'].detach().numpy(), cmap='RdBu_r', aspect='auto')
        axes[2].set_title('Annual Cycle Sine (L₁ˢ)')
        axes[2].set_xlabel('State Variable')
        axes[2].set_ylabel('State Variable')
        plt.colorbar(im3, ax=axes[2])
    
    # Semi-annual cycle cosine (L2_cos)
    if 'L2_cos' in components:
        im4 = axes[3].imshow(components['L2_cos'].detach().numpy(), cmap='RdBu_r', aspect='auto')
        axes[3].set_title('Semi-annual Cycle Cosine (L₂ᶜ)')
        axes[3].set_xlabel('State Variable')
        axes[3].set_ylabel('State Variable')
        plt.colorbar(im4, ax=axes[3])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # plt.show()  # Commented out to avoid popup during training


def main():
    parser = argparse.ArgumentParser(description='Evaluate Neural ODE models')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, choices=['neural_ode', 'physics_informed', 'stochastic'],
                       required=True, help='Type of model')
    parser.add_argument('--data_path', type=str, default='data/XRO_indices_oras5.nc',
                       help='Path to data file')
    parser.add_argument('--n_month', type=int, default=21,
                       help='Forecast horizon in months')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Auto-detect device if not specified
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = 'cuda'
            print(f'CUDA is available - using GPU: {torch.cuda.get_device_name(0)}')
        else:
            args.device = 'cpu'
            print('CUDA not available - using CPU')
    else:
        print(f'Using device: {args.device}')
    
    # Create results directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    print('Loading data...')
    obs_data = xr.open_dataset(args.data_path)
    var_names = list(obs_data.data_vars)
    
    # Split data for training XRO (same as in training)
    train_data = obs_data.sel(time=slice('1979-01', '2022-12'))
    
    # Load trained Neural ODE model
    print('Loading Neural ODE model...')
    neural_ode_model = load_trained_model(args.checkpoint, args.model_type, var_names, args.device)
    
    # Generate Neural ODE forecasts
    print('Generating Neural ODE forecasts...')
    neural_ode_forecasts = evaluate_forecast_skill(
        neural_ode_model, obs_data, args.n_month, args.device
    )
    
    # Create XRO model for comparison
    print('Setting up XRO model...')
    xro_model = XRO(ncycle=12, ac_order=2)
    
    # Compare with XRO
    print('Comparing with XRO...')
    results = compare_with_xro(neural_ode_forecasts, obs_data, xro_model, train_data)
    
    # Plot comparisons
    print('Plotting results...')
    
    # Forecast skill comparison
    plot_forecast_comparison(
        results, 
        var_name='Nino34',
        save_path=os.path.join(args.save_dir, 'forecast_skill_comparison.png')
    )
    
    # Forecast plume examples
    example_dates = ['1997-04', '2015-06', '2022-09']
    plot_forecast_plume(
        results,
        init_dates=example_dates,
        var_name='Nino34',
        obs_data=obs_data,
        save_path=os.path.join(args.save_dir, 'forecast_plume_comparison.png')
    )
    
    # Analyze model components (if physics-informed)
    if args.model_type == 'physics_informed':
        print('Analyzing model components...')
        analyze_model_components(
            neural_ode_model,
            save_path=os.path.join(args.save_dir, 'model_components.png')
        )
    
    # Print summary statistics
    print('\n' + '='*50)
    print('FORECAST SKILL SUMMARY')
    print('='*50)
    
    for var_name in ['Nino34', 'IOD', 'NPMM']:
        if var_name in results['neural_ode_skill']:
            neural_ode_6m = results['neural_ode_skill'][var_name].isel(lead=5).values
            xro_6m = results['xro_skill'][var_name].isel(lead=5).values
            
            neural_ode_12m = results['neural_ode_skill'][var_name].isel(lead=11).values
            xro_12m = results['xro_skill'][var_name].isel(lead=11).values
            
            print(f'\n{var_name}:')
            print(f'  6-month skill:  Neural ODE = {neural_ode_6m:.3f}, XRO = {xro_6m:.3f}')
            print(f'  12-month skill: Neural ODE = {neural_ode_12m:.3f}, XRO = {xro_12m:.3f}')
    
    print(f'\nResults saved to: {args.save_dir}')


if __name__ == '__main__':
    main()
