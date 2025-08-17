"""
Analyze oscillatory dynamics in XRO vs Neural ODE models

This script examines whether Neural ODE models exhibit oscillator-like behavior
similar to XRO's recharge oscillator dynamics.

Key analyses:
1. Eigenvalue analysis of linear operators
2. Phase portraits in state space
3. Spectral analysis of time series
4. Oscillation periods and damping rates
"""

import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import signal
from scipy.linalg import eigvals
# import seaborn as sns  # Not needed for this analysis
from model import NeuralODE, PhysicsInformedODE, StochasticNeuralODE
from xro.XRO import XRO
import os


def load_trained_model_for_analysis(model_type, var_names, device='cpu'):
    """Load a trained model for oscillatory analysis"""
    checkpoint_patterns = {
        'neural_ode': 'checkpoints/neural_ode_*.pt',
        'physics_informed': 'checkpoints/physics_informed_*.pt',
        'stochastic': 'checkpoints/stochastic_*.pt'
    }
    
    import glob
    checkpoint_files = glob.glob(checkpoint_patterns.get(model_type, ''))
    
    if not checkpoint_files:
        print(f"No checkpoint found for {model_type}, creating untrained model")
        state_dim = len(var_names)
        if model_type == 'neural_ode':
            return NeuralODE(state_dim=state_dim, var_names=var_names).to(device)
        elif model_type == 'physics_informed':
            return PhysicsInformedODE(state_dim=state_dim, var_names=var_names).to(device)
        elif model_type == 'stochastic':
            return StochasticNeuralODE(state_dim=state_dim, var_names=var_names, noise_scale=0.1).to(device)
    
    # Load the most recent checkpoint
    checkpoint_path = sorted(checkpoint_files)[-1]
    print(f"Loading {model_type} from {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dim = len(var_names)
        
        if model_type == 'neural_ode':
            model = NeuralODE(state_dim=state_dim, var_names=var_names)
        elif model_type == 'physics_informed':
            model = PhysicsInformedODE(state_dim=state_dim, var_names=var_names)
        elif model_type == 'stochastic':
            model = StochasticNeuralODE(state_dim=state_dim, var_names=var_names, noise_scale=0.1)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model
        
    except Exception as e:
        print(f"Failed to load {model_type}: {e}")
        return None


def analyze_xro_linear_operator(obs_data, train_data):
    """Analyze XRO's linear operator for oscillatory properties"""
    print("Analyzing XRO linear operator...")
    
    # Fit XRO model
    xro_model = XRO(ncycle=12, ac_order=2)
    xro_fit = xro_model.fit_matrix(train_data, maskb=['IOD'], maskNT=['T2', 'TH'])
    
    # Extract linear operator L (annual mean component)
    L_ac = xro_fit['Lac'].values  # [rank_y, rank_x, ncycle]
    L_mean = np.mean(L_ac, axis=2)  # Annual mean linear operator
    
    print(f"XRO Linear Operator L shape: {L_mean.shape}")
    print(f"XRO Linear Operator L:\n{L_mean}")
    
    # Compute eigenvalues
    eigenvals = eigvals(L_mean)
    print(f"XRO Eigenvalues: {eigenvals}")
    
    # Analyze oscillatory properties
    complex_eigs = eigenvals[np.iscomplex(eigenvals)]
    real_eigs = eigenvals[np.isreal(eigenvals)].real
    
    oscillatory_analysis = {
        'L_matrix': L_mean,
        'eigenvalues': eigenvals,
        'complex_eigenvalues': complex_eigs,
        'real_eigenvalues': real_eigs,
        'has_oscillatory_modes': len(complex_eigs) > 0,
        'damping_rates': -real_eigs,
        'oscillation_frequencies': np.abs(np.imag(complex_eigs)) if len(complex_eigs) > 0 else [],
        'oscillation_periods': 2*np.pi/np.abs(np.imag(complex_eigs)) if len(complex_eigs) > 0 else []
    }
    
    return oscillatory_analysis, xro_fit


def analyze_neural_ode_jacobian(model, x0, t=0.0):
    """Analyze Neural ODE's Jacobian matrix at a given state"""
    print(f"Analyzing Neural ODE Jacobian at state: {x0.flatten()}")
    
    # Convert to tensor and enable gradients
    x0_tensor = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
    t_tensor = torch.tensor([t], dtype=torch.float32)
    
    # Get the ODE function
    if hasattr(model, 'ode_func'):
        ode_func = model.ode_func
    else:
        # For models without explicit ode_func attribute
        return None
    
    # Compute dx/dt
    dxdt = ode_func(t_tensor, x0_tensor.unsqueeze(0)).squeeze(0)
    
    # Compute Jacobian matrix
    jacobian = torch.zeros(len(x0), len(x0))
    
    for i in range(len(dxdt)):
        # Compute gradient of dxdt[i] with respect to all x components
        grad_outputs = torch.zeros_like(dxdt)
        grad_outputs[i] = 1.0
        
        grads = torch.autograd.grad(
            outputs=dxdt, 
            inputs=x0_tensor, 
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=False
        )[0]
        
        jacobian[i, :] = grads
    
    jacobian_np = jacobian.detach().numpy()
    eigenvals = eigvals(jacobian_np)
    
    print(f"Neural ODE Jacobian:\n{jacobian_np}")
    print(f"Neural ODE Eigenvalues: {eigenvals}")
    
    # Analyze oscillatory properties
    complex_eigs = eigenvals[np.iscomplex(eigenvals)]
    real_eigs = eigenvals[np.isreal(eigenvals)].real
    
    jacobian_analysis = {
        'jacobian_matrix': jacobian_np,
        'eigenvalues': eigenvals,
        'complex_eigenvalues': complex_eigs,
        'real_eigenvalues': real_eigs,
        'has_oscillatory_modes': len(complex_eigs) > 0,
        'damping_rates': -real_eigs,
        'oscillation_frequencies': np.abs(np.imag(complex_eigs)) if len(complex_eigs) > 0 else [],
        'oscillation_periods': 2*np.pi/np.abs(np.imag(complex_eigs)) if len(complex_eigs) > 0 else []
    }
    
    return jacobian_analysis


def generate_phase_portraits(models, xro_fit, obs_data, var_names):
    """Generate phase portraits for XRO and Neural ODE models"""
    print("Generating phase portraits...")
    
    # Focus on first two variables for 2D phase portrait
    var1, var2 = var_names[0], var_names[1]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # XRO simulation
    print("Generating XRO simulation for phase portrait...")
    xro_model = XRO(ncycle=12, ac_order=2)
    x0_data = obs_data.isel(time=0)
    
    xro_sim = xro_model.simulate(
        fit_ds=xro_fit,
        x0_data=x0_data,
        nyear=5,
        ncopy=1,
        noise_type='red'
    )
    
    # Plot XRO phase portrait
    ax = axes[0, 0]
    ax.plot(xro_sim[var1].values.flatten(), xro_sim[var2].values.flatten(), 'b-', alpha=0.7)
    ax.scatter(xro_sim[var1].values[0], xro_sim[var2].values[0], color='green', s=50, label='Start')
    ax.scatter(xro_sim[var1].values[-1], xro_sim[var2].values[-1], color='red', s=50, label='End')
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.set_title('XRO Phase Portrait')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Neural ODE simulations
    model_names = ['neural_ode', 'physics_informed', 'stochastic']
    colors = ['red', 'blue', 'green']
    
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        if model_name in models and models[model_name] is not None:
            print(f"Generating {model_name} simulation for phase portrait...")
            
            model = models[model_name]
            
            with torch.no_grad():
                if model_name == 'stochastic':
                    sim = model.simulate(
                        x0_data=x0_data,
                        nyear=5,
                        ncopy=1,
                        enable_noise=True,
                        device='cpu'
                    )
                else:
                    sim = model.simulate(
                        x0_data=x0_data,
                        nyear=5,
                        ncopy=1,
                        add_noise=True,
                        device='cpu'
                    )
            
            # Plot Neural ODE phase portrait
            ax = axes[0, i+1] if i < 2 else axes[1, i-2]
            ax.plot(sim[var1].values.flatten(), sim[var2].values.flatten(), 
                   color=color, alpha=0.7, linewidth=1.5)
            ax.scatter(sim[var1].values[0], sim[var2].values[0], color='green', s=50, label='Start')
            ax.scatter(sim[var1].values[-1], sim[var2].values[-1], color='red', s=50, label='End')
            ax.set_xlabel(var1)
            ax.set_ylabel(var2)
            ax.set_title(f'{model_name.replace("_", " ").title()} Phase Portrait')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phase_portraits_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def spectral_analysis(models, xro_fit, obs_data, var_names):
    """Perform spectral analysis of time series from different models"""
    print("Performing spectral analysis...")
    
    # Focus on ENSO variable (Nino34)
    target_var = 'Nino34' if 'Nino34' in var_names else var_names[0]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # XRO spectral analysis
    print("XRO spectral analysis...")
    xro_model = XRO(ncycle=12, ac_order=2)
    x0_data = obs_data.isel(time=0)
    
    xro_sim = xro_model.simulate(
        fit_ds=xro_fit,
        x0_data=x0_data,
        nyear=10,
        ncopy=1,
        noise_type='red'
    )
    
    # Compute power spectral density for XRO
    xro_ts = xro_sim[target_var].values.flatten()
    freqs_xro, psd_xro = signal.periodogram(xro_ts, fs=12)  # 12 months per year
    
    ax = axes[0, 0]
    ax.loglog(freqs_xro[1:], psd_xro[1:], 'b-', linewidth=2, label='XRO')
    ax.axvline(1/3.5, color='red', linestyle='--', alpha=0.7, label='3.5-year ENSO')
    ax.axvline(1/2.5, color='orange', linestyle='--', alpha=0.7, label='2.5-year ENSO')
    ax.set_xlabel('Frequency (cycles/year)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title(f'XRO Spectrum ({target_var})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Neural ODE spectral analysis
    model_names = ['neural_ode', 'physics_informed', 'stochastic']
    colors = ['red', 'blue', 'green']
    ax_positions = [(0, 1), (1, 0), (1, 1)]
    
    for i, (model_name, color, ax_pos) in enumerate(zip(model_names, colors, ax_positions)):
        if model_name in models and models[model_name] is not None:
            print(f"{model_name} spectral analysis...")
            
            model = models[model_name]
            
            with torch.no_grad():
                if model_name == 'stochastic':
                    sim = model.simulate(
                        x0_data=x0_data,
                        nyear=10,
                        ncopy=1,
                        enable_noise=True,
                        device='cpu'
                    )
                else:
                    sim = model.simulate(
                        x0_data=x0_data,
                        nyear=10,
                        ncopy=1,
                        add_noise=True,
                        device='cpu'
                    )
            
            # Compute power spectral density
            neural_ts = sim[target_var].values.flatten()
            freqs_neural, psd_neural = signal.periodogram(neural_ts, fs=12)
            
            ax = axes[ax_pos]
            ax.loglog(freqs_neural[1:], psd_neural[1:], color=color, linewidth=2, 
                     label=model_name.replace('_', ' ').title())
            ax.axvline(1/3.5, color='red', linestyle='--', alpha=0.7, label='3.5-year ENSO')
            ax.axvline(1/2.5, color='orange', linestyle='--', alpha=0.7, label='2.5-year ENSO')
            ax.set_xlabel('Frequency (cycles/year)')
            ax.set_ylabel('Power Spectral Density')
            ax.set_title(f'{model_name.replace("_", " ").title()} Spectrum ({target_var})')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spectral_analysis_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_oscillatory_summary_table(xro_analysis, neural_analyses):
    """Create a summary table of oscillatory properties"""
    print("Creating oscillatory properties summary...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ['Model', 'Has Complex Eigenvalues', 'Oscillation Periods (years)', 'Damping Rates', 'Max Frequency (cycles/year)']
    
    # XRO analysis
    xro_periods = xro_analysis['oscillation_periods'] / 12 if len(xro_analysis['oscillation_periods']) > 0 else []
    xro_freqs = xro_analysis['oscillation_frequencies'] * 12 if len(xro_analysis['oscillation_frequencies']) > 0 else []
    
    table_data.append([
        'XRO',
        'Yes' if xro_analysis['has_oscillatory_modes'] else 'No',
        f"{xro_periods}" if len(xro_periods) > 0 else 'None',
        f"{xro_analysis['damping_rates']}" if len(xro_analysis['damping_rates']) > 0 else 'None',
        f"{max(xro_freqs):.3f}" if len(xro_freqs) > 0 else 'N/A'
    ])
    
    # Neural ODE analyses
    for model_name, analysis in neural_analyses.items():
        if analysis is not None:
            neural_periods = analysis['oscillation_periods'] / 12 if len(analysis['oscillation_periods']) > 0 else []
            neural_freqs = analysis['oscillation_frequencies'] * 12 if len(analysis['oscillation_frequencies']) > 0 else []
            
            table_data.append([
                model_name.replace('_', ' ').title(),
                'Yes' if analysis['has_oscillatory_modes'] else 'No',
                f"{neural_periods}" if len(neural_periods) > 0 else 'None',
                f"{analysis['damping_rates']}" if len(analysis['damping_rates']) > 0 else 'None',
                f"{max(neural_freqs):.3f}" if len(neural_freqs) > 0 else 'N/A'
            ])
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Color code the table
    colors = ['lightgray', 'lightcoral', 'lightblue', 'lightgreen']
    for i, color in enumerate(colors[:len(table_data)]):
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(color)
    
    ax.set_title('Oscillatory Properties Comparison\n(Complex eigenvalues indicate oscillatory behavior)', 
                 fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('oscillatory_properties_table.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main analysis function"""
    print("="*80)
    print("OSCILLATORY DYNAMICS ANALYSIS")
    print("Comparing XRO vs Neural ODE oscillator-like behavior")
    print("="*80)
    
    # Load data
    print("Loading data...")
    obs_data = xr.open_dataset('data/XRO_indices_oras5.nc')
    var_names = list(obs_data.data_vars)
    train_data = obs_data.sel(time=slice('1979-01', '2022-12'))
    
    print(f"Variables: {var_names}")
    
    # Analyze XRO oscillatory properties
    print("\n" + "="*60)
    print("ANALYZING XRO OSCILLATORY PROPERTIES")
    print("="*60)
    
    xro_analysis, xro_fit = analyze_xro_linear_operator(obs_data, train_data)
    
    print(f"\nXRO Oscillatory Analysis:")
    print(f"  Has oscillatory modes: {xro_analysis['has_oscillatory_modes']}")
    if xro_analysis['has_oscillatory_modes']:
        periods_years = xro_analysis['oscillation_periods'] / 12
        print(f"  Oscillation periods: {periods_years} years")
        print(f"  Oscillation frequencies: {xro_analysis['oscillation_frequencies']} rad/month")
    
    # Load Neural ODE models
    print("\n" + "="*60)
    print("LOADING NEURAL ODE MODELS")
    print("="*60)
    
    models = {}
    model_types = ['neural_ode', 'physics_informed', 'stochastic']
    
    for model_type in model_types:
        model = load_trained_model_for_analysis(model_type, var_names)
        if model is not None:
            models[model_type] = model
            print(f"✓ Loaded {model_type}")
        else:
            print(f"✗ Failed to load {model_type}")
    
    # Analyze Neural ODE Jacobians
    print("\n" + "="*60)
    print("ANALYZING NEURAL ODE JACOBIANS")
    print("="*60)
    
    neural_analyses = {}
    
    # Use a typical ENSO state for analysis
    x0_typical = np.array([1.0, 0.5, -0.3, 0.2])[:len(var_names)]  # Typical ENSO state
    
    for model_name, model in models.items():
        print(f"\nAnalyzing {model_name}...")
        try:
            analysis = analyze_neural_ode_jacobian(model, x0_typical)
            neural_analyses[model_name] = analysis
            
            if analysis and analysis['has_oscillatory_modes']:
                periods_years = analysis['oscillation_periods'] / 12
                print(f"  ✓ {model_name} has oscillatory modes!")
                print(f"    Oscillation periods: {periods_years} years")
            else:
                print(f"  ✗ {model_name} lacks oscillatory modes")
                
        except Exception as e:
            print(f"  ✗ Failed to analyze {model_name}: {e}")
            neural_analyses[model_name] = None
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Phase portraits
    generate_phase_portraits(models, xro_fit, obs_data, var_names)
    
    # Spectral analysis
    spectral_analysis(models, xro_fit, obs_data, var_names)
    
    # Summary table
    create_oscillatory_summary_table(xro_analysis, neural_analyses)
    
    # Final summary
    print("\n" + "="*80)
    print("OSCILLATORY ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nXRO Oscillatory Behavior:")
    print(f"  ✓ Has complex eigenvalues: {xro_analysis['has_oscillatory_modes']}")
    if xro_analysis['has_oscillatory_modes']:
        periods = xro_analysis['oscillation_periods'] / 12
        print(f"  ✓ Oscillation periods: {periods} years")
    
    print(f"\nNeural ODE Oscillatory Behavior:")
    oscillatory_models = []
    non_oscillatory_models = []
    
    for model_name, analysis in neural_analyses.items():
        if analysis and analysis['has_oscillatory_modes']:
            oscillatory_models.append(model_name)
            periods = analysis['oscillation_periods'] / 12
            print(f"  ✓ {model_name}: Has oscillatory modes (periods: {periods} years)")
        else:
            non_oscillatory_models.append(model_name)
            print(f"  ✗ {model_name}: No oscillatory modes detected")
    
    print(f"\n📊 Results saved:")
    print(f"  - phase_portraits_comparison.png")
    print(f"  - spectral_analysis_comparison.png") 
    print(f"  - oscillatory_properties_table.png")
    
    if len(non_oscillatory_models) > 0:
        print(f"\n⚠️  Models lacking oscillatory behavior: {non_oscillatory_models}")
        print(f"   Consider implementing oscillator constraints or physics-informed losses!")
    
    if len(oscillatory_models) > 0:
        print(f"\n✅ Models with oscillatory behavior: {oscillatory_models}")


if __name__ == '__main__':
    main()
