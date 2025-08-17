"""
Simple oscillatory dynamics analysis for XRO vs Neural ODE models

Focus on the key question: Do Neural ODE models exhibit oscillator-like behavior?
"""

import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.linalg import eigvals
from model import NeuralODE, PhysicsInformedODE, StochasticNeuralODE
from xro.XRO import XRO


def analyze_xro_oscillatory_properties(obs_data, train_data):
    """Analyze XRO's linear operator for oscillatory properties"""
    print("Analyzing XRO oscillatory properties...")
    
    # Fit XRO model
    xro_model = XRO(ncycle=12, ac_order=2)
    xro_fit = xro_model.fit_matrix(train_data, maskb=['IOD'], maskNT=['T2', 'TH'])
    
    # Extract linear operator L (annual mean component)
    L_ac = xro_fit['Lac'].values  # [rank_y, rank_x, ncycle]
    L_mean = np.mean(L_ac, axis=2)  # Annual mean linear operator
    
    print(f"XRO Linear Operator L shape: {L_mean.shape}")
    
    # Compute eigenvalues
    eigenvals = eigvals(L_mean)
    
    # Analyze oscillatory properties
    complex_eigs = eigenvals[np.iscomplex(eigenvals)]
    real_eigs = eigenvals[np.isreal(eigenvals)].real
    
    print(f"XRO Eigenvalues: {eigenvals}")
    print(f"Complex eigenvalues (oscillatory): {len(complex_eigs)}")
    print(f"Real eigenvalues (non-oscillatory): {len(real_eigs)}")
    
    if len(complex_eigs) > 0:
        frequencies = np.abs(np.imag(complex_eigs))
        periods_months = 2*np.pi/frequencies
        periods_years = periods_months / 12
        print(f"Oscillation periods: {periods_years} years")
        
        # Check for ENSO-like periods (2-7 years)
        enso_periods = periods_years[(periods_years >= 2) & (periods_years <= 7)]
        print(f"ENSO-like periods (2-7 years): {enso_periods}")
    
    return {
        'L_matrix': L_mean,
        'eigenvalues': eigenvals,
        'has_oscillatory_modes': len(complex_eigs) > 0,
        'n_oscillatory_modes': len(complex_eigs),
        'oscillation_periods_years': periods_years if len(complex_eigs) > 0 else []
    }


def create_simple_neural_ode(state_dim, var_names):
    """Create a simple Neural ODE for analysis"""
    return NeuralODE(state_dim=state_dim, var_names=var_names, hidden_dim=32)


def analyze_neural_ode_at_equilibrium(model, device='cpu'):
    """Analyze Neural ODE Jacobian at equilibrium (zero state)"""
    print("Analyzing Neural ODE at equilibrium state...")
    
    state_dim = len(model.var_names)
    
    # Use equilibrium state (zeros)
    x0 = torch.zeros(1, state_dim, dtype=torch.float32, requires_grad=True, device=device)
    t = torch.tensor([0.0], dtype=torch.float32, device=device)
    
    # Get the ODE function
    try:
        # Compute dx/dt at equilibrium
        dxdt = model.ode_func(t, x0)
        
        # Compute Jacobian matrix using autograd
        jacobian = torch.zeros(state_dim, state_dim, device=device)
        
        for i in range(state_dim):
            # Zero out gradients
            if x0.grad is not None:
                x0.grad.zero_()
            
            # Compute gradient of dxdt[i] with respect to x
            dxdt_i = dxdt[0, i]
            dxdt_i.backward(retain_graph=True)
            
            if x0.grad is not None:
                jacobian[i, :] = x0.grad[0, :].clone()
        
        jacobian_np = jacobian.detach().cpu().numpy()
        eigenvals_j = eigvals(jacobian_np)
        
        print(f"Neural ODE Jacobian shape: {jacobian_np.shape}")
        print(f"Neural ODE Eigenvalues: {eigenvals_j}")
        
        # Analyze oscillatory properties
        complex_eigs = eigenvals_j[np.iscomplex(eigenvals_j)]
        real_eigs = eigenvals_j[np.isreal(eigenvals_j)].real
        
        print(f"Complex eigenvalues (oscillatory): {len(complex_eigs)}")
        print(f"Real eigenvalues (non-oscillatory): {len(real_eigs)}")
        
        if len(complex_eigs) > 0:
            frequencies = np.abs(np.imag(complex_eigs))
            periods_months = 2*np.pi/frequencies
            periods_years = periods_months / 12
            print(f"Oscillation periods: {periods_years} years")
        
        return {
            'jacobian_matrix': jacobian_np,
            'eigenvalues': eigenvals_j,
            'has_oscillatory_modes': len(complex_eigs) > 0,
            'n_oscillatory_modes': len(complex_eigs),
            'oscillation_periods_years': periods_years if len(complex_eigs) > 0 else []
        }
        
    except Exception as e:
        print(f"Error analyzing Neural ODE: {e}")
        return None


def plot_eigenvalue_comparison(xro_analysis, neural_analysis):
    """Plot eigenvalues in complex plane"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # XRO eigenvalues
    xro_eigs = xro_analysis['eigenvalues']
    ax1.scatter(np.real(xro_eigs), np.imag(xro_eigs), c='blue', s=50, alpha=0.7, label='XRO')
    ax1.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(0, color='k', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Real Part')
    ax1.set_ylabel('Imaginary Part')
    ax1.set_title('XRO Eigenvalues')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add unit circle for reference
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit Circle')
    
    # Neural ODE eigenvalues
    if neural_analysis is not None:
        neural_eigs = neural_analysis['eigenvalues']
        ax2.scatter(np.real(neural_eigs), np.imag(neural_eigs), c='red', s=50, alpha=0.7, label='Neural ODE')
        ax2.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax2.axvline(0, color='k', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Real Part')
        ax2.set_ylabel('Imaginary Part')
        ax2.set_title('Neural ODE Eigenvalues')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add unit circle for reference
        ax2.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit Circle')
    else:
        ax2.text(0.5, 0.5, 'Neural ODE Analysis Failed', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Neural ODE Eigenvalues (Failed)')
    
    plt.tight_layout()
    plt.savefig('eigenvalue_comparison.png', dpi=300, bbox_inches='tight')
    # plt.show()  # Commented out to avoid popup during training


def create_oscillatory_summary(xro_analysis, neural_analysis):
    """Create summary of oscillatory properties"""
    print("\n" + "="*80)
    print("OSCILLATORY BEHAVIOR SUMMARY")
    print("="*80)
    
    print(f"\n🔵 XRO Model:")
    print(f"   Has oscillatory modes: {xro_analysis['has_oscillatory_modes']}")
    print(f"   Number of oscillatory modes: {xro_analysis['n_oscillatory_modes']}")
    if xro_analysis['has_oscillatory_modes']:
        periods = xro_analysis['oscillation_periods_years']
        enso_periods = periods[(periods >= 2) & (periods <= 7)]
        print(f"   Oscillation periods: {periods} years")
        print(f"   ENSO-like periods (2-7 years): {enso_periods}")
    
    print(f"\n🔴 Neural ODE Model:")
    if neural_analysis is not None:
        print(f"   Has oscillatory modes: {neural_analysis['has_oscillatory_modes']}")
        print(f"   Number of oscillatory modes: {neural_analysis['n_oscillatory_modes']}")
        if neural_analysis['has_oscillatory_modes']:
            periods = neural_analysis['oscillation_periods_years']
            enso_periods = periods[(periods >= 2) & (periods <= 7)]
            print(f"   Oscillation periods: {periods} years")
            print(f"   ENSO-like periods (2-7 years): {enso_periods}")
    else:
        print(f"   Analysis failed - could not compute Jacobian")
    
    # Diagnosis and recommendations
    print(f"\n📊 DIAGNOSIS:")
    
    if xro_analysis['has_oscillatory_modes']:
        print(f"   ✅ XRO exhibits oscillator-like behavior (as expected)")
    else:
        print(f"   ❌ XRO lacks oscillatory modes (unexpected!)")
    
    if neural_analysis is not None:
        if neural_analysis['has_oscillatory_modes']:
            print(f"   ✅ Neural ODE exhibits oscillator-like behavior")
            
            # Compare periods
            if xro_analysis['has_oscillatory_modes']:
                xro_periods = xro_analysis['oscillation_periods_years']
                neural_periods = neural_analysis['oscillation_periods_years']
                
                xro_enso = xro_periods[(xro_periods >= 2) & (xro_periods <= 7)]
                neural_enso = neural_periods[(neural_periods >= 2) & (neural_periods <= 7)]
                
                if len(neural_enso) > 0:
                    print(f"   ✅ Neural ODE has ENSO-like oscillations")
                else:
                    print(f"   ⚠️  Neural ODE oscillations are not ENSO-like")
        else:
            print(f"   ❌ Neural ODE lacks oscillator-like behavior")
            print(f"   🔧 RECOMMENDATION: Add oscillator constraints!")
    else:
        print(f"   ❌ Could not analyze Neural ODE oscillatory behavior")
        print(f"   🔧 RECOMMENDATION: Fix model architecture issues")


def suggest_oscillator_improvements():
    """Suggest ways to improve oscillatory behavior in Neural ODE"""
    print(f"\n" + "="*80)
    print("RECOMMENDATIONS FOR OSCILLATORY BEHAVIOR")
    print("="*80)
    
    print(f"""
🎯 If Neural ODE lacks oscillatory behavior, consider these approaches:

1. 📐 PHYSICS-INFORMED CONSTRAINTS:
   - Add oscillator equation terms: d²x/dt² + ω²x = f(x,t)
   - Include recharge oscillator dynamics explicitly
   - Constrain eigenvalues to have imaginary components

2. 🎛️ LOSS FUNCTION MODIFICATIONS:
   - Add spectral loss: penalize lack of 2-7 year periods
   - Add phase space loss: encourage circular/elliptical trajectories
   - Add eigenvalue regularization: encourage complex eigenvalues

3. 🏗️ ARCHITECTURAL CHANGES:
   - Use coupled oscillator networks
   - Add explicit harmonic oscillator components
   - Include seasonal forcing terms

4. 📊 TRAINING MODIFICATIONS:
   - Train on oscillatory data specifically
   - Use curriculum learning: start with simple oscillations
   - Add oscillatory initial conditions

5. 🔄 HYBRID APPROACHES:
   - Combine Neural ODE with analytical oscillator
   - Use Neural ODE for nonlinear terms, analytical for linear oscillator
   - Implement learnable oscillator parameters
""")


def main():
    """Main analysis function"""
    print("="*80)
    print("SIMPLE OSCILLATORY DYNAMICS ANALYSIS")
    print("XRO vs Neural ODE Oscillator Behavior")
    print("="*80)
    
    # Load data
    print("Loading data...")
    obs_data = xr.open_dataset('data/XRO_indices_oras5.nc')
    var_names = list(obs_data.data_vars)
    train_data = obs_data.sel(time=slice('1979-01', '2022-12'))
    
    print(f"Variables: {var_names}")
    print(f"State dimension: {len(var_names)}")
    
    # Analyze XRO
    print("\n" + "="*60)
    print("ANALYZING XRO")
    print("="*60)
    
    xro_analysis = analyze_xro_oscillatory_properties(obs_data, train_data)
    
    # Create and analyze Neural ODE
    print("\n" + "="*60)
    print("ANALYZING NEURAL ODE")
    print("="*60)
    
    print("Creating untrained Neural ODE...")
    neural_model = create_simple_neural_ode(len(var_names), var_names)
    neural_model.eval()
    
    neural_analysis = analyze_neural_ode_at_equilibrium(neural_model)
    
    # Create visualizations
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    plot_eigenvalue_comparison(xro_analysis, neural_analysis)
    
    # Summary and recommendations
    create_oscillatory_summary(xro_analysis, neural_analysis)
    suggest_oscillator_improvements()
    
    print(f"\n📁 Output files:")
    print(f"   - eigenvalue_comparison.png")


if __name__ == '__main__':
    main()
