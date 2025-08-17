"""
Test script for OscillatorNeuralODE variants (external vs internal noise)
"""

import torch
import numpy as np
import xarray as xr
from model import OscillatorNeuralODE


def test_oscillator_variants():
    """Test both external and internal noise variants"""
    print("Testing OscillatorNeuralODE variants...")
    
    var_names = ['Nino34', 'WWV', 'NPMM', 'IOD']
    state_dim = len(var_names)
    
    # Test external noise variant
    print("\n1. Testing External Noise Variant:")
    model_external = OscillatorNeuralODE(
        state_dim=state_dim, 
        var_names=var_names, 
        noise_mode='external'
    )
    
    print(f"   Noise mode: {model_external.noise_mode}")
    print(f"   Has noise_net: {hasattr(model_external, 'noise_net')}")
    print(f"   ODE func type: {type(model_external.ode_func).__name__}")
    
    # Test internal noise variant
    print("\n2. Testing Internal Noise Variant:")
    model_internal = OscillatorNeuralODE(
        state_dim=state_dim, 
        var_names=var_names, 
        noise_mode='internal',
        noise_scale=0.1
    )
    
    print(f"   Noise mode: {model_internal.noise_mode}")
    print(f"   Noise scale: {model_internal.noise_scale}")
    print(f"   ODE func type: {type(model_internal.ode_func).__name__}")
    print(f"   ODE func has noise_scale: {hasattr(model_internal.ode_func, 'noise_scale')}")
    
    # Test forward passes
    print("\n3. Testing Forward Passes:")
    x0 = torch.randn(2, state_dim)
    t = torch.linspace(0, 12, 13)
    
    with torch.no_grad():
        # External model
        sol_ext_no_noise = model_external.forward(x0, t, add_noise=False)
        sol_ext_with_noise = model_external.forward(x0, t, add_noise=True)
        
        print(f"   External - No noise: {sol_ext_no_noise.shape}")
        print(f"   External - With noise: {sol_ext_with_noise.shape}")
        
        # Internal model
        sol_int_no_noise = model_internal.forward(x0, t, enable_noise=False)
        sol_int_with_noise = model_internal.forward(x0, t, enable_noise=True)
        
        print(f"   Internal - No noise: {sol_int_no_noise.shape}")
        print(f"   Internal - With noise: {sol_int_with_noise.shape}")
    
    # Test simulation interface
    print("\n4. Testing Simulation Interface:")
    
    # Create dummy initial conditions
    x0_data = xr.Dataset({
        var: xr.DataArray([0.5 * np.random.randn()], dims=['dummy'])
        for var in var_names
    })
    
    # External model simulation
    sim_ext = model_external.simulate(x0_data, nyear=1, ncopy=2, add_noise=True)
    print(f"   External simulation: {list(sim_ext.dims.values())}")
    
    # Internal model simulation
    sim_int = model_internal.simulate(x0_data, nyear=1, ncopy=2, enable_noise=True)
    print(f"   Internal simulation: {list(sim_int.dims.values())}")
    
    # Test reforecast interface
    print("\n5. Testing Reforecast Interface:")
    
    # Create dummy initialization data
    time_coord = xr.date_range(start='2000-01', periods=5, freq='MS')
    init_data = xr.Dataset({
        var: xr.DataArray(
            np.random.randn(5) * 0.5,
            coords={'time': time_coord},
            dims=['time']
        )
        for var in var_names
    })
    
    # External model reforecast
    fcst_ext = model_external.reforecast(init_data, n_month=6, ncopy=1, add_noise=False)
    print(f"   External reforecast: {list(fcst_ext.dims.values())}")
    
    # Internal model reforecast
    fcst_int = model_internal.reforecast(init_data, n_month=6, ncopy=1, enable_noise=False)
    print(f"   Internal reforecast: {list(fcst_int.dims.values())}")
    
    # Test oscillator info
    print("\n6. Testing Oscillator Info:")
    
    osc_info_ext = model_external.get_oscillator_info()
    osc_info_int = model_internal.get_oscillator_info()
    
    print(f"   External periods: {osc_info_ext['periods_years']} years")
    print(f"   Internal periods: {osc_info_int['periods_years']} years")
    
    print("\n✅ All OscillatorNeuralODE variants working correctly!")
    
    return model_external, model_internal


if __name__ == '__main__':
    test_oscillator_variants()
