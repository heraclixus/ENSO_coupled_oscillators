"""
Test script for Stochastic Neural ODE model
"""

import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from model import StochasticNeuralODE


def test_stochastic_functionality():
    """Test basic stochastic model functionality"""
    print("Testing Stochastic Neural ODE functionality...")
    
    # Load data
    data = xr.open_dataset('data/XRO_indices_oras5.nc')
    var_names = list(data.data_vars)
    state_dim = len(var_names)
    
    print(f"Data variables: {var_names}")
    print(f"State dimension: {state_dim}")
    
    # Test Stochastic Neural ODE
    print("\n1. Testing StochasticNeuralODE...")
    stochastic_ode = StochasticNeuralODE(
        state_dim=state_dim, 
        var_names=var_names,
        noise_scale=0.1
    )
    
    # Test forward pass with noise
    x0 = torch.randn(2, state_dim)  # 2 samples
    t = torch.linspace(0, 1, 13)    # 1 year, monthly
    
    print("\n2. Testing deterministic mode...")
    with torch.no_grad():
        output_det = stochastic_ode(x0, t, enable_noise=False)
    
    print(f"Deterministic output shape: {output_det.shape}")
    print("✓ Deterministic mode successful")
    
    print("\n3. Testing stochastic mode...")
    with torch.no_grad():
        output_stoc = stochastic_ode(x0, t, enable_noise=True)
    
    print(f"Stochastic output shape: {output_stoc.shape}")
    print("✓ Stochastic mode successful")
    
    # Check that stochastic and deterministic outputs are different
    diff = torch.abs(output_stoc - output_det).mean()
    print(f"Mean difference between stochastic and deterministic: {diff:.6f}")
    
    if diff > 1e-6:
        print("✓ Stochastic noise is being applied correctly")
    else:
        print("⚠️  Warning: Stochastic and deterministic outputs are very similar")
    
    return stochastic_ode


def test_noise_characteristics(model):
    """Test noise characteristics over time"""
    print("\n4. Testing noise characteristics...")
    
    # Analyze noise over a full year
    t_points = torch.linspace(0, 1, 12)  # Monthly points over one year
    
    with torch.no_grad():
        noise_stats = model.get_noise_characteristics(t_points)
    
    print(f"Noise std shape: {noise_stats['noise_std'].shape}")
    print(f"Seasonal amplitude shape: {noise_stats['seasonal_amplitude'].shape}")
    
    # Plot noise characteristics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot seasonal noise amplitude for first 4 variables
    for i in range(min(4, model.state_dim)):
        ax = axes.flatten()[i]
        
        seasonal_amp = noise_stats['seasonal_amplitude'][:, i].numpy()
        noise_std = noise_stats['noise_std'][:, i].numpy()
        
        months = np.arange(1, 13)
        ax.plot(months, seasonal_amp, 'b-', linewidth=2, label='Seasonal Amplitude')
        ax.fill_between(months, seasonal_amp - noise_std, seasonal_amp + noise_std, 
                       alpha=0.3, color='blue', label='±1 Std')
        
        ax.set_title(f'{model.var_names[i]} Noise Characteristics')
        ax.set_xlabel('Month')
        ax.set_ylabel('Noise Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stochastic_noise_characteristics.png', dpi=150, bbox_inches='tight')
    # plt.show()  # Commented out to avoid popup during training
    
    print("✓ Noise characteristics analysis completed")


def test_ensemble_simulation(model):
    """Test ensemble simulation with stochastic model"""
    print("\n5. Testing ensemble simulation...")
    
    # Load data
    data = xr.open_dataset('data/XRO_indices_oras5.nc')
    x0_data = data.isel(time=0)
    
    # Test stochastic simulation
    with torch.no_grad():
        stoc_sim = model.simulate(
            x0_data=x0_data,
            nyear=2,
            ncopy=10,
            enable_noise=True,
            device='cpu'
        )
        
        det_sim = model.simulate(
            x0_data=x0_data,
            nyear=2,
            ncopy=1,
            enable_noise=False,
            device='cpu'
        )
    
    print(f"Stochastic simulation shape: {dict(stoc_sim.dims)}")
    print(f"Deterministic simulation shape: {dict(det_sim.dims)}")
    
    # Plot ensemble spread
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    variables_to_plot = ['Nino34', 'WWV', 'IOD', 'NPMM']
    
    for i, var in enumerate(variables_to_plot):
        ax = axes[i]
        
        # Plot ensemble members (first 5)
        for member in range(min(5, stoc_sim.member.size)):
            stoc_sim[var].isel(member=member).plot(
                ax=ax, alpha=0.5, linewidth=1, color='red'
            )
        
        # Plot ensemble mean
        stoc_sim[var].mean('member').plot(
            ax=ax, color='red', linewidth=2, label='Stochastic (Ens. Mean)'
        )
        
        # Plot deterministic
        det_sim[var].isel(member=0).plot(
            ax=ax, color='blue', linewidth=2, label='Deterministic'
        )
        
        ax.set_title(f'{var} - Stochastic vs Deterministic')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stochastic_ensemble_comparison.png', dpi=150, bbox_inches='tight')
    # plt.show()  # Commented out to avoid popup during training
    
    print("✓ Ensemble simulation completed")
    
    return stoc_sim, det_sim


def test_forecast_comparison(model):
    """Test forecasting with stochastic model"""
    print("\n6. Testing stochastic forecasting...")
    
    # Load data
    data = xr.open_dataset('data/XRO_indices_oras5.nc')
    init_data = data.isel(time=slice(100, 105))  # 5 initialization times
    
    with torch.no_grad():
        stoc_forecast = model.reforecast(
            init_data=init_data,
            n_month=12,
            ncopy=10,
            enable_noise=True,
            device='cpu'
        )
        
        det_forecast = model.reforecast(
            init_data=init_data,
            n_month=12,
            ncopy=1,
            enable_noise=False,
            device='cpu'
        )
    
    print(f"Stochastic forecast shape: {dict(stoc_forecast.dims)}")
    print(f"Deterministic forecast shape: {dict(det_forecast.dims)}")
    
    # Plot forecast comparison for one initialization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    init_idx = 2  # Choose middle initialization
    variables_to_plot = ['Nino34', 'WWV', 'IOD', 'NPMM']
    
    for i, var in enumerate(variables_to_plot):
        ax = axes[i]
        
        # Plot stochastic ensemble
        for member in range(min(5, stoc_forecast.member.size)):
            stoc_forecast[var].isel(init=init_idx, member=member).plot(
                ax=ax, alpha=0.5, linewidth=1, color='red'
            )
        
        # Plot ensemble mean
        stoc_forecast[var].isel(init=init_idx).mean('member').plot(
            ax=ax, color='red', linewidth=2, label='Stochastic (Ens. Mean)'
        )
        
        # Plot deterministic forecast
        det_forecast[var].isel(init=init_idx).plot(
            ax=ax, color='blue', linewidth=2, label='Deterministic'
        )
        
        ax.set_title(f'{var} Forecast Comparison')
        ax.set_xlabel('Lead (months)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stochastic_forecast_comparison.png', dpi=150, bbox_inches='tight')
    # plt.show()  # Commented out to avoid popup during training
    
    print("✓ Stochastic forecasting completed")


def test_noise_scaling(model):
    """Test different noise scales"""
    print("\n7. Testing noise scaling effects...")
    
    # Load data
    data = xr.open_dataset('data/XRO_indices_oras5.nc')
    x0_data = data.isel(time=0)
    
    noise_scales = [0.0, 0.05, 0.1, 0.2]
    simulations = {}
    
    with torch.no_grad():
        for scale in noise_scales:
            sim = model.simulate(
                x0_data=x0_data,
                nyear=1,
                ncopy=5,
                enable_noise=True,
                noise_scale=scale,
                device='cpu'
            )
            simulations[scale] = sim
    
    # Plot effect of different noise scales
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    variables_to_plot = ['Nino34', 'WWV', 'IOD', 'NPMM']
    
    for i, var in enumerate(variables_to_plot):
        ax = axes[i]
        
        for scale in noise_scales:
            sim = simulations[scale]
            # Plot ensemble mean
            sim[var].mean('member').plot(
                ax=ax, linewidth=2, label=f'Noise Scale = {scale}'
            )
        
        ax.set_title(f'{var} - Effect of Noise Scale')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stochastic_noise_scaling.png', dpi=150, bbox_inches='tight')
    # plt.show()  # Commented out to avoid popup during training
    
    print("✓ Noise scaling test completed")


def main():
    """Main test function"""
    print("="*60)
    print("STOCHASTIC NEURAL ODE MODEL TESTING")
    print("="*60)
    
    try:
        # Basic functionality tests
        model = test_stochastic_functionality()
        
        # Noise characteristics
        test_noise_characteristics(model)
        
        # Ensemble simulation
        stoc_sim, det_sim = test_ensemble_simulation(model)
        
        # Forecast comparison
        test_forecast_comparison(model)
        
        # Noise scaling effects
        test_noise_scaling(model)
        
        print("\n" + "="*60)
        print("ALL STOCHASTIC TESTS PASSED SUCCESSFULLY! ✓")
        print("="*60)
        
        print(f"\nFiles created:")
        print(f"- stochastic_noise_characteristics.png")
        print(f"- stochastic_ensemble_comparison.png")
        print(f"- stochastic_forecast_comparison.png")
        print(f"- stochastic_noise_scaling.png")
        
        print(f"\nNext steps:")
        print(f"1. Train stochastic model: python train.py --model_type stochastic --n_epochs 50")
        print(f"2. Compare performance with other models")
        print(f"3. Analyze learned noise patterns")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
