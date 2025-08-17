"""
Simple ENSO forecast skill comparison focusing on trained models
Creates plots showing ENSO forecast performance with working skill metrics

This is the RECOMMENDED comparison script - it works reliably and focuses on ENSO.
For advanced comparison (currently broken), see trained_model_comparison_advanced.py
"""

import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from model import NeuralODE, PhysicsInformedODE, StochasticNeuralODE, OscillatorNeuralODE, GraphNeuralODE, PhysicsGraphNeuralODE, SineGraphNeuralODE, SinePhysicsGraphNeuralODE
from xro.XRO import XRO
import os
import glob
from typing import Dict, List
from scipy.stats import pearsonr


def find_checkpoint_files(checkpoint_dir='checkpoints'):
    """Find checkpoint files and map them to model types"""
    if not os.path.exists(checkpoint_dir):
        print(f"Warning: Checkpoint directory '{checkpoint_dir}' not found!")
        return {}
    
    # Look for checkpoint files in both root directory and subdirectories
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, '*.pt'))  # Root level (legacy)
    checkpoint_files.extend(glob.glob(os.path.join(checkpoint_dir, '*', '*.pt')))  # Subdirectories (new structure)
    checkpoint_map = {}
    
    for file_path in checkpoint_files:
        filename = os.path.basename(file_path)
        
        if 'neural_ode_enso_only' in filename.lower():
            checkpoint_map['NODE_external_ENSO'] = file_path
        elif 'neural_ode' in filename.lower():
            checkpoint_map['NODE_external'] = file_path
        elif 'physics_informed_enso_only' in filename.lower():
            checkpoint_map['PhysicsNODE_external_ENSO'] = file_path
        elif 'physics_informed' in filename.lower():
            checkpoint_map['PhysicsNODE_external'] = file_path
        elif 'stochastic_enso_only' in filename.lower():
            checkpoint_map['NODE_internal_ENSO'] = file_path
        elif 'stochastic' in filename.lower():
            checkpoint_map['NODE_internal'] = file_path
        elif 'oscillator_external_enso_only' in filename.lower():
            checkpoint_map['OscillatorNODE_external_ENSO'] = file_path
        elif 'oscillator_external' in filename.lower():
            checkpoint_map['OscillatorNODE_external'] = file_path
        elif 'oscillator_internal_enso_only' in filename.lower():
            checkpoint_map['OscillatorNODE_internal_ENSO'] = file_path
        elif 'oscillator_internal' in filename.lower():
            checkpoint_map['OscillatorNODE_internal'] = file_path
        elif 'graph_gcn_enso_only' in filename.lower():
            checkpoint_map['GraphNODE_GCN_external_ENSO'] = file_path
        elif 'graph_gcn' in filename.lower():
            checkpoint_map['GraphNODE_GCN_external'] = file_path
        elif 'graph_gat_enso_only' in filename.lower():
            checkpoint_map['GraphNODE_GAT_external_ENSO'] = file_path
        elif 'graph_gat' in filename.lower():
            checkpoint_map['GraphNODE_GAT_external'] = file_path
        elif 'physics_graph_gcn_enso_only' in filename.lower():
            checkpoint_map['PhysicsGraphNODE_GCN_external_ENSO'] = file_path
        elif 'physics_graph_gcn' in filename.lower():
            checkpoint_map['PhysicsGraphNODE_GCN_external'] = file_path
        elif 'physics_graph_gat_enso_only' in filename.lower():
            checkpoint_map['PhysicsGraphNODE_GAT_external_ENSO'] = file_path
        elif 'physics_graph_gat' in filename.lower():
            checkpoint_map['PhysicsGraphNODE_GAT_external'] = file_path
        elif 'sine_graph_sine_enso_only' in filename.lower():
            checkpoint_map['SineGraphNODE_Sine_external_ENSO'] = file_path
        elif 'sine_graph_sine' in filename.lower():
            checkpoint_map['SineGraphNODE_Sine_external'] = file_path
        elif 'sine_graph_kuramoto_enso_only' in filename.lower():
            checkpoint_map['SineGraphNODE_Kuramoto_external_ENSO'] = file_path
        elif 'sine_graph_kuramoto' in filename.lower():
            checkpoint_map['SineGraphNODE_Kuramoto_external'] = file_path
        elif 'sine_physics_graph_sine_enso_only' in filename.lower():
            checkpoint_map['SinePhysicsGraphNODE_Sine_external_ENSO'] = file_path
        elif 'sine_physics_graph_sine' in filename.lower():
            checkpoint_map['SinePhysicsGraphNODE_Sine_external'] = file_path
        elif 'sine_physics_graph_kuramoto_enso_only' in filename.lower():
            checkpoint_map['SinePhysicsGraphNODE_Kuramoto_external_ENSO'] = file_path
        elif 'sine_physics_graph_kuramoto' in filename.lower():
            checkpoint_map['SinePhysicsGraphNODE_Kuramoto_external'] = file_path
    
    return checkpoint_map


def load_trained_model(checkpoint_path: str, model_type: str, var_names: List[str], device='cpu'):
    """Load a trained model from checkpoint with proper hidden_dim"""
    print(f"Loading {model_type} from {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dim = len(var_names)
        
        # Use hidden_dim=64 to match the saved checkpoints
        if model_type in ['NODE_external', 'NODE_external_ENSO']:
            model = NeuralODE(state_dim=state_dim, var_names=var_names, hidden_dim=64)
        elif model_type in ['NODE_internal', 'NODE_internal_ENSO']:
            model = StochasticNeuralODE(state_dim=state_dim, var_names=var_names, noise_scale=0.1, hidden_dim=64)
        elif model_type in ['PhysicsNODE_external', 'PhysicsNODE_external_ENSO']:
            model = PhysicsInformedODE(state_dim=state_dim, var_names=var_names, hidden_dim=64)
        elif model_type in ['OscillatorNODE_external', 'OscillatorNODE_external_ENSO']:
            model = OscillatorNeuralODE(state_dim=state_dim, var_names=var_names, hidden_dim=64, noise_mode='external')
        elif model_type in ['OscillatorNODE_internal', 'OscillatorNODE_internal_ENSO']:
            model = OscillatorNeuralODE(state_dim=state_dim, var_names=var_names, hidden_dim=64, noise_mode='internal')
        elif model_type in ['GraphNODE_GCN_external', 'GraphNODE_GCN_external_ENSO']:
            model = GraphNeuralODE(state_dim=state_dim, var_names=var_names, hidden_dim=64, gnn_type='gcn', noise_mode='external')
        elif model_type in ['GraphNODE_GAT_external', 'GraphNODE_GAT_external_ENSO']:
            model = GraphNeuralODE(state_dim=state_dim, var_names=var_names, hidden_dim=64, gnn_type='gat', noise_mode='external')
        elif model_type in ['PhysicsGraphNODE_GCN_external', 'PhysicsGraphNODE_GCN_external_ENSO']:
            model = PhysicsGraphNeuralODE(state_dim=state_dim, var_names=var_names, hidden_dim=64, gnn_type='gcn', noise_mode='external')
        elif model_type in ['PhysicsGraphNODE_GAT_external', 'PhysicsGraphNODE_GAT_external_ENSO']:
            model = PhysicsGraphNeuralODE(state_dim=state_dim, var_names=var_names, hidden_dim=64, gnn_type='gat', noise_mode='external')
        elif model_type in ['SineGraphNODE_Sine_external', 'SineGraphNODE_Sine_external_ENSO']:
            model = SineGraphNeuralODE(state_dim=state_dim, var_names=var_names, hidden_dim=64, sine_type='sine', noise_mode='external')
        elif model_type in ['SineGraphNODE_Kuramoto_external', 'SineGraphNODE_Kuramoto_external_ENSO']:
            model = SineGraphNeuralODE(state_dim=state_dim, var_names=var_names, hidden_dim=64, sine_type='kuramoto', noise_mode='external')
        elif model_type in ['SinePhysicsGraphNODE_Sine_external', 'SinePhysicsGraphNODE_Sine_external_ENSO']:
            model = SinePhysicsGraphNeuralODE(state_dim=state_dim, var_names=var_names, hidden_dim=64, sine_type='sine', noise_mode='external')
        elif model_type in ['SinePhysicsGraphNODE_Kuramoto_external', 'SinePhysicsGraphNODE_Kuramoto_external_ENSO']:
            model = SinePhysicsGraphNeuralODE(state_dim=state_dim, var_names=var_names, hidden_dim=64, sine_type='kuramoto', noise_mode='external')
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"✓ Successfully loaded {model_type}")
        return model
        
    except Exception as e:
        print(f"✗ Failed to load {model_type}: {e}")
        return None


def simple_correlation_skill(forecast: xr.Dataset, obs: xr.Dataset, var_name: str = 'Nino34') -> np.ndarray:
    """
    Compute simple correlation skill for each lead time
    
    Args:
        forecast: Forecast dataset with dims (init, lead)
        obs: Observation dataset with time dimension
        var_name: Variable to compute skill for
    
    Returns:
        Array of correlation skills for each lead time
    """
    n_leads = forecast.sizes['lead']
    correlations = np.full(n_leads, np.nan)
    
    for lead in range(n_leads):
        # Get forecast values for this lead time
        fcst_values = forecast[var_name].isel(lead=lead).values
        
        # Get corresponding observation values
        # For each init time, get the observation at init_time + lead
        obs_values = []
        valid_fcst = []
        
        for i, init_time in enumerate(forecast.init.values):
            # Find the target time (init + lead months)
            try:
                # Convert to pandas timestamp for easier manipulation
                init_pd = pd.Timestamp(init_time)
                target_time = init_pd + pd.DateOffset(months=lead)
                
                # Find closest observation time
                obs_time_idx = np.argmin(np.abs(obs.time.values - np.datetime64(target_time)))
                obs_val = obs[var_name].isel(time=obs_time_idx).values
                
                if not np.isnan(fcst_values[i]) and not np.isnan(obs_val):
                    valid_fcst.append(fcst_values[i])
                    obs_values.append(obs_val)
                    
            except:
                continue
        
        # Compute correlation if we have enough valid pairs
        if len(valid_fcst) > 5:
            corr, _ = pearsonr(valid_fcst, obs_values)
            correlations[lead] = corr
    
    return correlations


def generate_simple_forecasts(models, obs_data, device='cpu'):
    """Generate forecasts for all models using a simple approach"""
    print("\nGenerating forecasts for all models...")
    
    forecasts = {}
    
    # Use a subset of data for faster computation
    subset_data = obs_data.isel(time=slice(100, 200))  # 100 time points
    
    # 1. XRO forecasts
    print("\n1. Generating XRO forecasts...")
    try:
        xro_model = XRO(ncycle=12, ac_order=2)
        train_data = obs_data.sel(time=slice('1979-01', '2022-12'))
        xro_fit = xro_model.fit_matrix(train_data, maskb=['IOD'], maskNT=['T2', 'TH'])
        
        xro_forecasts = xro_model.reforecast(
            fit_ds=xro_fit,
            init_ds=subset_data,
            n_month=12,
            ncopy=1,
            noise_type='zero'
        )
        forecasts['XRO'] = xro_forecasts
        print("✓ Successfully generated XRO forecasts")
    except Exception as e:
        print(f"✗ Failed to generate XRO forecasts: {e}")
    
    # 2. Neural ODE variants
    for i, (name, model) in enumerate(models.items(), 2):
        print(f"\n{i}. Generating {name} forecasts...")
        
        if model is None:
            print(f"✗ Skipping {name} - model not loaded")
            continue
            
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            try:
                if 'internal' in name:
                    # Internal noise models (StochasticNeuralODE and OscillatorNeuralODE internal)
                    forecast = model.reforecast(
                        init_data=subset_data,
                        n_month=12,
                        ncopy=1,
                        enable_noise=False,
                        device=device
                    )
                else:
                    # External noise models (NeuralODE, PhysicsInformedODE, OscillatorNeuralODE external)
                    forecast = model.reforecast(
                        init_data=subset_data,
                        n_month=12,
                        ncopy=1,
                        add_noise=False,
                        device=device
                    )
                
                forecasts[name] = forecast
                print(f"✓ Successfully generated {name} forecasts")
                
            except Exception as e:
                print(f"✗ Failed to generate {name} forecasts: {e}")
    
    return forecasts, subset_data


def compute_simple_skills(forecasts, obs_data):
    """Compute simple correlation skills for ENSO"""
    print("\nComputing ENSO forecast skills...")
    
    skills = {}
    
    for model_name, forecast in forecasts.items():
        print(f"Computing skills for {model_name}...")
        
        try:
            # Compute correlation skill for ENSO
            corr_skill = simple_correlation_skill(forecast, obs_data, 'Nino34')
            skills[model_name] = corr_skill
            
            # Print some stats
            valid_skills = corr_skill[~np.isnan(corr_skill)]
            if len(valid_skills) > 0:
                print(f"  ✓ {model_name}: Mean skill = {np.mean(valid_skills):.3f}")
            else:
                print(f"  ✗ {model_name}: No valid skills computed")
                
        except Exception as e:
            print(f"✗ Failed to compute skills for {model_name}: {e}")
            skills[model_name] = np.full(13, np.nan)  # 13 lead times (0-12)
    
    return skills


def plot_enso_skills(skills, save_path='enso_forecast_skills.png'):
    """Plot ENSO forecast skills"""
    print("\nCreating ENSO forecast skill plots...")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Define colors and styles for each model
    model_styles = {
        'XRO': {'color': 'black', 'linestyle': '-', 'linewidth': 3, 'marker': 'o', 'markersize': 6},
        'NODE_external': {'color': 'red', 'linestyle': '-', 'linewidth': 2, 'marker': 's', 'markersize': 5},
        'NODE_external_ENSO': {'color': 'red', 'linestyle': ':', 'linewidth': 2, 'marker': 'D', 'markersize': 4},
        'NODE_internal': {'color': 'red', 'linestyle': '--', 'linewidth': 2, 'marker': '^', 'markersize': 5},
        'NODE_internal_ENSO': {'color': 'red', 'linestyle': '-.', 'linewidth': 2, 'marker': 'v', 'markersize': 4},
        'PhysicsNODE_external': {'color': 'blue', 'linestyle': '-', 'linewidth': 2, 'marker': 's', 'markersize': 5},
        'PhysicsNODE_external_ENSO': {'color': 'blue', 'linestyle': ':', 'linewidth': 2, 'marker': 'D', 'markersize': 4},
        'PhysicsNODE_internal': {'color': 'blue', 'linestyle': '--', 'linewidth': 2, 'marker': '^', 'markersize': 5},
        'PhysicsNODE_internal_ENSO': {'color': 'blue', 'linestyle': '-.', 'linewidth': 2, 'marker': 'v', 'markersize': 4},
        'OscillatorNODE_external': {'color': 'green', 'linestyle': '-', 'linewidth': 2, 'marker': 's', 'markersize': 5},
        'OscillatorNODE_external_ENSO': {'color': 'green', 'linestyle': ':', 'linewidth': 2, 'marker': 'D', 'markersize': 4},
        'OscillatorNODE_internal': {'color': 'green', 'linestyle': '--', 'linewidth': 2, 'marker': '^', 'markersize': 5},
        'OscillatorNODE_internal_ENSO': {'color': 'green', 'linestyle': '-.', 'linewidth': 2, 'marker': 'v', 'markersize': 4},
        'GraphNODE_GCN_external': {'color': 'purple', 'linestyle': '-', 'linewidth': 2, 'marker': 's', 'markersize': 5},
        'GraphNODE_GCN_external_ENSO': {'color': 'purple', 'linestyle': ':', 'linewidth': 2, 'marker': 'D', 'markersize': 4},
        'GraphNODE_GAT_external': {'color': 'purple', 'linestyle': '--', 'linewidth': 2, 'marker': '^', 'markersize': 5},
        'GraphNODE_GAT_external_ENSO': {'color': 'purple', 'linestyle': '-.', 'linewidth': 2, 'marker': 'v', 'markersize': 4},
        'PhysicsGraphNODE_GCN_external': {'color': 'orange', 'linestyle': '-', 'linewidth': 2, 'marker': 's', 'markersize': 5},
        'PhysicsGraphNODE_GCN_external_ENSO': {'color': 'orange', 'linestyle': ':', 'linewidth': 2, 'marker': 'D', 'markersize': 4},
        'PhysicsGraphNODE_GAT_external': {'color': 'orange', 'linestyle': '--', 'linewidth': 2, 'marker': '^', 'markersize': 5},
        'PhysicsGraphNODE_GAT_external_ENSO': {'color': 'orange', 'linestyle': '-.', 'linewidth': 2, 'marker': 'v', 'markersize': 4},
        'SineGraphNODE_Sine_external': {'color': 'brown', 'linestyle': '-', 'linewidth': 2, 'marker': 's', 'markersize': 5},
        'SineGraphNODE_Sine_external_ENSO': {'color': 'brown', 'linestyle': ':', 'linewidth': 2, 'marker': 'D', 'markersize': 4},
        'SineGraphNODE_Kuramoto_external': {'color': 'brown', 'linestyle': '--', 'linewidth': 2, 'marker': '^', 'markersize': 5},
        'SineGraphNODE_Kuramoto_external_ENSO': {'color': 'brown', 'linestyle': '-.', 'linewidth': 2, 'marker': 'v', 'markersize': 4},
        'SinePhysicsGraphNODE_Sine_external': {'color': 'pink', 'linestyle': '-', 'linewidth': 2, 'marker': 's', 'markersize': 5},
        'SinePhysicsGraphNODE_Sine_external_ENSO': {'color': 'pink', 'linestyle': ':', 'linewidth': 2, 'marker': 'D', 'markersize': 4},
        'SinePhysicsGraphNODE_Kuramoto_external': {'color': 'pink', 'linestyle': '--', 'linewidth': 2, 'marker': '^', 'markersize': 5},
        'SinePhysicsGraphNODE_Kuramoto_external_ENSO': {'color': 'pink', 'linestyle': '-.', 'linewidth': 2, 'marker': 'v', 'markersize': 4}
    }
    
    # Plot skills
    for model_name, skill_values in skills.items():
        if not np.all(np.isnan(skill_values)):
            style = model_styles.get(model_name, {'color': 'gray', 'linestyle': '-'})
            lead_times = np.arange(len(skill_values))
            ax.plot(lead_times, skill_values, label=model_name, **style)
    
    ax.set_title('ENSO (Nino3.4) Forecast Correlation Skill')
    ax.set_ylabel('Correlation')
    ax.set_xlabel('Lead (months)')
    ax.axhline(0.5, ls=':', color='gray', alpha=0.7, label='0.5 threshold')
    ax.axhline(0.0, ls='-', color='gray', alpha=0.3)
    ax.set_ylim([-0.2, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to: {save_path}")


def main():
    """Main comparison function"""
    print("="*80)
    print("COMPREHENSIVE ENSO FORECAST SKILL COMPARISON")
    print("Models: XRO, Neural ODE, Physics NODE, Stochastic NODE, Oscillator NODE, Graph NODE, Physics Graph NODE, Sine Graph NODE, Sine Physics Graph NODE")
    print("Variants: External/Internal noise, ENSO-only/Multivariate training, GCN/GAT layers, Sine/Kuramoto aggregation")
    print("="*80)
    
    # Load data
    print("Loading data...")
    obs_data = xr.open_dataset('data/XRO_indices_oras5.nc')
    var_names = list(obs_data.data_vars)
    
    print(f"Variables: {var_names}")
    
    # Find and load trained models
    print("\nLoading trained models...")
    checkpoint_map = find_checkpoint_files('checkpoints')
    print(f"Found checkpoints: {list(checkpoint_map.keys())}")
    
    models = {}
    for model_type, checkpoint_path in checkpoint_map.items():
        model = load_trained_model(checkpoint_path, model_type, var_names)
        if model is not None:
            models[model_type] = model
    
    if not models:
        print("\n❌ No models loaded! Please train models first.")
        return
    
    print(f"\nLoaded models: {list(models.keys())}")
    
    # Generate forecasts
    forecasts, subset_data = generate_simple_forecasts(models, obs_data)
    
    if len(forecasts) <= 1:
        print("\n❌ Not enough forecasts generated!")
        return
    
    # Compute skills
    skills = compute_simple_skills(forecasts, obs_data)
    
    # Plot results
    plot_enso_skills(skills, 'enso_forecast_skills_simple.png')
    
    # Print summary
    print("\n" + "="*80)
    print("ENSO FORECAST SKILL SUMMARY")
    print("="*80)
    
    print("\nForecast Skills at different lead times:")
    for model_name, skill_values in skills.items():
        if not np.all(np.isnan(skill_values)):
            # 3-month, 6-month, 12-month skills
            skill_3m = skill_values[3] if len(skill_values) > 3 else np.nan
            skill_6m = skill_values[6] if len(skill_values) > 6 else np.nan
            skill_12m = skill_values[12] if len(skill_values) > 12 else skill_values[-1]
            
            print(f"  {model_name:25}: 3m={skill_3m:.3f}, 6m={skill_6m:.3f}, 12m={skill_12m:.3f}")
    
    print(f"\nComparison Complete!")
    print(f"Generated plot: enso_forecast_skills_simple.png")


if __name__ == '__main__':
    # Import pandas for date manipulation
    import pandas as pd
    main()
