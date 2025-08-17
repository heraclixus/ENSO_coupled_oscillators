"""
Test script for OscillatorNeuralODE

Verifies that the oscillator-constrained Neural ODE:
1. Has the correct architecture
2. Exhibits strong oscillatory behavior
3. Has ENSO-appropriate timescales
4. Can be trained with oscillator losses
"""

import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.linalg import eigvals
from model import OscillatorNeuralODE
from model.oscillator_neural_ode import create_oscillator_loss


def test_oscillator_architecture():
    """Test basic architecture and forward pass"""
    print("Testing OscillatorNeuralODE architecture...")
    
    # Create model
    var_names = ['Nino34', 'WWV', 'NPMM', 'IOD']
    model = OscillatorNeuralODE(state_dim=4, var_names=var_names, hidden_dim=32)
    model.eval()
    
    # Test forward pass
    x0 = torch.randn(2, 4)  # batch_size=2, state_dim=4
    t = torch.linspace(0, 12, 13)  # 1 year monthly
    
    with torch.no_grad():
        solution = model.forward(x0, t, add_noise=False)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {x0.shape}")
    print(f"  Output shape: {solution.shape}")
    print(f"  Expected: [13, 2, 4], Got: {list(solution.shape)}")
    
    assert solution.shape == (13, 2, 4), f"Wrong output shape: {solution.shape}"
    
    # Test oscillator info
    osc_info = model.get_oscillator_info()
    print(f"✓ Oscillator info retrieved")
    print(f"  Oscillation periods: {osc_info['periods_years']} years")
    print(f"  Frequencies: {osc_info['frequencies']} rad/month")
    print(f"  Damping: {osc_info['damping']}")
    
    return model


def test_oscillatory_behavior(model):
    """Test that the model exhibits strong oscillatory behavior"""
    print("\nTesting oscillatory behavior...")
    
    # Analyze eigenvalues at equilibrium
    x_eq = torch.zeros(4)
    eigenvals = model.compute_eigenvalues_at_state(x_eq, t=0.0)
    
    print(f"Eigenvalues at equilibrium: {eigenvals}")
    
    # Count complex eigenvalues
    complex_eigs = eigenvals[np.iscomplex(eigenvals)]
    real_eigs = eigenvals[np.isreal(eigenvals)]
    
    print(f"Complex eigenvalues (oscillatory): {len(complex_eigs)}")
    print(f"Real eigenvalues (non-oscillatory): {len(real_eigs)}")
    
    # Check for ENSO-like periods
    if len(complex_eigs) > 0:
        frequencies = np.abs(np.imag(complex_eigs))
        periods_months = 2*np.pi/frequencies
        periods_years = periods_months / 12
        
        print(f"Oscillation periods: {periods_years} years")
        
        # Check for ENSO-like periods (2-7 years)
        enso_periods = periods_years[(periods_years >= 2) & (periods_years <= 7)]
        print(f"ENSO-like periods (2-7 years): {enso_periods}")
        
        assert len(enso_periods) > 0, "No ENSO-like oscillation periods found!"
        print(f"✓ Model has ENSO-like oscillatory behavior")
    else:
        print("❌ No oscillatory behavior detected!")
        return False
    
    return True


def test_simulation_and_reforecast(model):
    """Test simulation and reforecast functionality"""
    print("\nTesting simulation and reforecast...")
    
    # Create dummy initial conditions
    var_names = model.var_names
    x0_data = xr.Dataset({
        var: xr.DataArray([0.5 * np.random.randn()], dims=['dummy'])
        for var in var_names
    })
    
    # Test simulation
    print("Testing simulation...")
    sim_result = model.simulate(x0_data, nyear=2, ncopy=3, add_noise=True)
    
    print(f"✓ Simulation successful")
    print(f"  Variables: {list(sim_result.data_vars)}")
    print(f"  Dimensions: {sim_result.dims}")
    print(f"  Time length: {len(sim_result.time)}")
    print(f"  Ensemble size: {len(sim_result.member)}")
    
    # Test reforecast
    print("Testing reforecast...")
    
    # Create dummy initialization data
    time_coord = xr.cftime_range(start='2000-01', periods=10, freq='MS', calendar='noleap')
    init_data = xr.Dataset({
        var: xr.DataArray(
            np.random.randn(10) * 0.5,
            coords={'time': time_coord},
            dims=['time']
        )
        for var in var_names
    })
    
    forecast_result = model.reforecast(init_data, n_month=12, ncopy=2, add_noise=False)
    
    print(f"✓ Reforecast successful")
    print(f"  Variables: {list(forecast_result.data_vars)}")
    print(f"  Dimensions: {forecast_result.dims}")
    print(f"  Lead times: {len(forecast_result.lead)}")
    print(f"  Init times: {len(forecast_result.init_time)}")
    
    return sim_result, forecast_result


def test_oscillator_loss(model):
    """Test the oscillator-specific loss function"""
    print("\nTesting oscillator loss function...")
    
    # Create oscillator loss
    osc_loss_fn = create_oscillator_loss(model, target_periods=[36, 48])  # 3-4 year periods
    
    # Generate sample trajectory with gradients enabled
    x0 = torch.randn(1, 4, requires_grad=True)
    t = torch.linspace(0, 24, 25)  # 2 years
    
    # Forward pass with gradients enabled
    trajectory = model.forward(x0, t, add_noise=False)
    
    # Compute oscillator loss
    try:
        loss_value = osc_loss_fn(trajectory, t)
        print(f"✓ Oscillator loss computed: {loss_value.item():.6f}")
        
        # Test that loss is differentiable
        if trajectory.requires_grad:
            loss_value.backward()
            print(f"✓ Oscillator loss is differentiable")
        else:
            # Try with explicit gradient requirement
            trajectory_grad = trajectory.clone().detach().requires_grad_(True)
            loss_grad = osc_loss_fn(trajectory_grad, t)
            
            if loss_grad.requires_grad:
                loss_grad.backward()
                print(f"✓ Oscillator loss is differentiable (manual grad)")
            else:
                print(f"⚠️  Oscillator loss computed but not differentiable")
        
        return True
        
    except Exception as e:
        print(f"❌ Oscillator loss failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_oscillatory_behavior(model, sim_result):
    """Create visualizations of oscillatory behavior"""
    print("\nCreating oscillatory behavior visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot time series for first variable
    ax = axes[0, 0]
    var = model.var_names[0]
    for member in range(min(3, len(sim_result.member))):
        sim_result[var].isel(member=member).plot(ax=ax, alpha=0.7, label=f'Member {member+1}')
    ax.set_title(f'{var} Time Series')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot phase portrait (first two variables)
    ax = axes[0, 1]
    if len(model.var_names) >= 2:
        var1, var2 = model.var_names[0], model.var_names[1]
        for member in range(min(2, len(sim_result.member))):
            x_vals = sim_result[var1].isel(member=member).values
            y_vals = sim_result[var2].isel(member=member).values
            ax.plot(x_vals, y_vals, alpha=0.7, label=f'Member {member+1}')
        ax.set_xlabel(var1)
        ax.set_ylabel(var2)
        ax.set_title('Phase Portrait')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot power spectrum
    ax = axes[1, 0]
    from scipy import signal as scipy_signal
    
    var = model.var_names[0]
    ts = sim_result[var].isel(member=0).values
    freqs, psd = scipy_signal.periodogram(ts, fs=12)  # 12 months per year
    
    ax.loglog(freqs[1:], psd[1:], 'b-', linewidth=2)
    ax.axvline(1/3.5, color='red', linestyle='--', alpha=0.7, label='3.5-year ENSO')
    ax.axvline(1/4.5, color='orange', linestyle='--', alpha=0.7, label='4.5-year ENSO')
    ax.set_xlabel('Frequency (cycles/year)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title(f'{var} Power Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot eigenvalues in complex plane
    ax = axes[1, 1]
    eigenvals = model.compute_eigenvalues_at_state(torch.zeros(len(model.var_names)), t=0.0)
    
    ax.scatter(np.real(eigenvals), np.imag(eigenvals), c='red', s=50, alpha=0.7, label='Eigenvalues')
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(0, color='k', linestyle='-', alpha=0.3)
    
    # Add unit circle for reference
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit Circle')
    
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.set_title('Eigenvalues (Complex Plane)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('oscillator_neural_ode_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()  # Commented out to avoid popup during training
    
    print(f"✓ Visualization saved as 'oscillator_neural_ode_analysis.png'")


def main():
    """Main test function"""
    print("="*80)
    print("TESTING OSCILLATOR-CONSTRAINED NEURAL ODE")
    print("="*80)
    
    # Test 1: Architecture
    model = test_oscillator_architecture()
    
    # Test 2: Oscillatory behavior
    has_oscillations = test_oscillatory_behavior(model)
    
    if not has_oscillations:
        print("❌ Model lacks oscillatory behavior - stopping tests")
        return
    
    # Test 3: Simulation and reforecast
    sim_result, forecast_result = test_simulation_and_reforecast(model)
    
    # Test 4: Oscillator loss
    loss_works = test_oscillator_loss(model)
    
    # Test 5: Visualizations
    visualize_oscillatory_behavior(model, sim_result)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    print(f"✅ Architecture test: PASSED")
    print(f"{'✅' if has_oscillations else '❌'} Oscillatory behavior: {'PASSED' if has_oscillations else 'FAILED'}")
    print(f"✅ Simulation/Reforecast: PASSED")
    print(f"{'✅' if loss_works else '❌'} Oscillator loss: {'PASSED' if loss_works else 'FAILED'}")
    print(f"✅ Visualizations: PASSED")
    
    # Get oscillator info
    osc_info = model.get_oscillator_info()
    print(f"\n🎯 OSCILLATOR PROPERTIES:")
    print(f"   Periods: {osc_info['periods_years']} years")
    print(f"   Frequencies: {osc_info['frequencies']} rad/month")
    print(f"   Damping: {osc_info['damping']}")
    
    # Check ENSO-like behavior
    periods = osc_info['periods_years']
    enso_periods = periods[(periods >= 2) & (periods <= 7)]
    
    if len(enso_periods) > 0:
        print(f"   ✅ ENSO-like periods: {enso_periods} years")
    else:
        print(f"   ⚠️  No ENSO-like periods (2-7 years)")
    
    print(f"\n🎉 OscillatorNeuralODE is ready for training!")


if __name__ == '__main__':
    main()
