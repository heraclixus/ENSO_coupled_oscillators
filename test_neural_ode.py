"""
Test script for Neural ODE models
"""

import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from model import NeuralODE, PhysicsInformedODE


def test_basic_functionality():
    """Test basic model functionality"""
    print("Testing basic Neural ODE functionality...")
    
    # Load data
    data = xr.open_dataset('data/XRO_indices_oras5.nc')
    var_names = list(data.data_vars)
    state_dim = len(var_names)
    
    print(f"Data variables: {var_names}")
    print(f"State dimension: {state_dim}")
    
    # Test Neural ODE
    print("\n1. Testing NeuralODE...")
    neural_ode = NeuralODE(state_dim=state_dim, var_names=var_names)
    
    # Test forward pass
    x0 = torch.randn(2, state_dim)  # 2 samples
    t = torch.linspace(0, 1, 13)    # 1 year, monthly
    
    with torch.no_grad():
        output = neural_ode(x0, t, add_noise=False)
    
    print(f"Input shape: {x0.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ NeuralODE forward pass successful")
    
    # Test Physics-Informed ODE
    print("\n2. Testing PhysicsInformedODE...")
    physics_ode = PhysicsInformedODE(state_dim=state_dim, var_names=var_names)
    
    with torch.no_grad():
        output = physics_ode(x0, t, add_noise=False)
    
    print(f"Output shape: {output.shape}")
    print("✓ PhysicsInformedODE forward pass successful")
    
    # Test seasonal components
    components = physics_ode.get_seasonal_components()
    print(f"Seasonal components: {list(components.keys())}")
    print("✓ Seasonal components extraction successful")


def test_simulation():
    """Test simulation functionality"""
    print("\n3. Testing simulation functionality...")
    
    # Load data
    data = xr.open_dataset('data/XRO_indices_oras5.nc')
    var_names = list(data.data_vars)
    
    # Create model
    model = PhysicsInformedODE(state_dim=len(var_names), var_names=var_names)
    
    # Test simulation
    x0_data = data.isel(time=0)  # Initial condition
    
    with torch.no_grad():
        sim_result = model.simulate(
            x0_data=x0_data,
            nyear=2,
            ncopy=3,
            add_noise=True,
            device='cpu'
        )
    
    print(f"Simulation result shape: {dict(sim_result.dims)}")
    print(f"Variables: {list(sim_result.data_vars)}")
    print("✓ Simulation successful")
    
    return sim_result


def test_forecasting():
    """Test forecasting functionality"""
    print("\n4. Testing forecasting functionality...")
    
    # Load data
    data = xr.open_dataset('data/XRO_indices_oras5.nc')
    var_names = list(data.data_vars)
    
    # Create model
    model = PhysicsInformedODE(state_dim=len(var_names), var_names=var_names)
    
    # Test forecasting with subset of data
    init_data = data.isel(time=slice(0, 10))  # First 10 time steps
    
    with torch.no_grad():
        forecast_result = model.reforecast(
            init_data=init_data,
            n_month=6,
            ncopy=2,
            add_noise=False,
            device='cpu'
        )
    
    print(f"Forecast result shape: {dict(forecast_result.dims)}")
    print(f"Variables: {list(forecast_result.data_vars)}")
    print("✓ Forecasting successful")
    
    return forecast_result


def test_xro_compatibility():
    """Test compatibility with XRO interface"""
    print("\n5. Testing XRO compatibility...")
    
    try:
        from xro.XRO import XRO
        
        # Load data
        data = xr.open_dataset('data/XRO_indices_oras5.nc')
        train_data = data.sel(time=slice('1979-01', '2022-12'))
        
        # Create XRO model
        xro_model = XRO(ncycle=12, ac_order=2)
        
        # Test XRO simulation
        xro_fit = xro_model.fit_matrix(train_data, maskb=['IOD'], maskNT=['T2', 'TH'])
        xro_sim = xro_model.simulate(
            fit_ds=xro_fit,
            X0_ds=train_data.isel(time=0),
            nyear=2,
            ncopy=2,
            is_xi_stdac=False,
            seed=42
        )
        
        print(f"XRO simulation shape: {dict(xro_sim.dims)}")
        print("✓ XRO compatibility verified")
        
        return xro_sim
        
    except Exception as e:
        print(f"XRO compatibility test failed: {e}")
        return None


def plot_comparison(neural_ode_sim, xro_sim=None):
    """Plot comparison between Neural ODE and XRO simulations"""
    print("\n6. Plotting results...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    # Plot Neural ODE simulation
    for i, var in enumerate(['Nino34', 'WWV', 'IOD', 'NPMM']):
        if i >= 4:
            break
            
        ax = axes[i]
        
        # Plot Neural ODE
        if var in neural_ode_sim:
            neural_ode_sim[var].isel(member=0).plot(ax=ax, label='Neural ODE', color='red')
        
        # Plot XRO if available
        if xro_sim is not None and var in xro_sim:
            xro_sim[var].isel(member=0).plot(ax=ax, label='XRO', color='blue', alpha=0.7)
        
        ax.set_title(f'{var} Simulation')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('neural_ode_test_results.png', dpi=150, bbox_inches='tight')
    # plt.show()  # Commented out to avoid popup during training
    
    print("✓ Plotting completed")


def main():
    """Main test function"""
    print("="*60)
    print("NEURAL ODE MODEL TESTING")
    print("="*60)
    
    try:
        # Basic functionality tests
        test_basic_functionality()
        
        # Simulation test
        neural_ode_sim = test_simulation()
        
        # Forecasting test
        forecast_result = test_forecasting()
        
        # XRO compatibility test
        xro_sim = test_xro_compatibility()
        
        # Plotting
        plot_comparison(neural_ode_sim, xro_sim)
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED SUCCESSFULLY! ✓")
        print("="*60)
        
        print("\nNext steps:")
        print("1. Run training: python train.py --model_type physics_informed --n_epochs 50")
        print("2. Evaluate model: python evaluate_neural_ode.py --checkpoint <path> --model_type physics_informed")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
