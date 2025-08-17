"""
ADVANCED forecast skill comparison using TRAINED Neural ODE models vs XRO
Loads trained models from checkpoints and compares forecast skills

⚠️  WARNING: This script currently has climpred API compatibility issues.
    Use simple_enso_comparison.py for reliable ENSO-focused comparison.
    
This script is kept for future development when climpred issues are resolved.
"""

import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from model import NeuralODE, PhysicsInformedODE, StochasticNeuralODE
from xro.XRO import XRO
from utils import calc_forecast_skill
import os
import glob
from typing import Dict, List, Optional


def find_checkpoint_files(checkpoint_dir='checkpoints'):
    """
    Find checkpoint files and map them to model types based on filename patterns
    
    Expected naming convention:
    - neural_ode_*.pt -> NODE_external
    - neural_ode_enso_only_*.pt -> NODE_external_ENSO
    - physics_informed_*.pt -> PhysicsNODE_external  
    - physics_informed_enso_only_*.pt -> PhysicsNODE_external_ENSO
    - stochastic_*.pt -> NODE_internal (or PhysicsNODE_internal)
    - stochastic_enso_only_*.pt -> NODE_internal_ENSO
    """
    if not os.path.exists(checkpoint_dir):
        print(f"Warning: Checkpoint directory '{checkpoint_dir}' not found!")
        return {}
    
    # Look for checkpoint files in both root directory and subdirectories
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, '*.pt'))  # Root level (legacy)
    checkpoint_files.extend(glob.glob(os.path.join(checkpoint_dir, '*', '*.pt')))  # Subdirectories (new structure)
    
    checkpoint_map = {}
    
    for file_path in checkpoint_files:
        filename = os.path.basename(file_path)
        
        # Map filenames to model types based on naming convention
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
            # For now, map stochastic to NODE_internal
            # Could be enhanced to distinguish physics vs non-physics stochastic
            checkpoint_map['NODE_internal'] = file_path
    
    return checkpoint_map


def load_trained_model(checkpoint_path: str, model_type: str, var_names: List[str], device='cpu'):
    """Load a trained model from checkpoint"""
    print(f"Loading {model_type} from {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dim = len(var_names)
        
        # Create model based on type
        if model_type == 'NODE_external':
            model = NeuralODE(state_dim=state_dim, var_names=var_names)
        elif model_type == 'NODE_external_ENSO':
            model = NeuralODE(state_dim=state_dim, var_names=var_names)
        elif model_type == 'NODE_internal':
            model = StochasticNeuralODE(state_dim=state_dim, var_names=var_names, noise_scale=0.1)
        elif model_type == 'NODE_internal_ENSO':
            model = StochasticNeuralODE(state_dim=state_dim, var_names=var_names, noise_scale=0.1)
        elif model_type == 'PhysicsNODE_external':
            model = PhysicsInformedODE(state_dim=state_dim, var_names=var_names)
        elif model_type == 'PhysicsNODE_external_ENSO':
            model = PhysicsInformedODE(state_dim=state_dim, var_names=var_names)
        elif model_type == 'PhysicsNODE_internal':
            model = StochasticNeuralODE(state_dim=state_dim, var_names=var_names, noise_scale=0.1)
        elif model_type == 'PhysicsNODE_internal_ENSO':
            model = StochasticNeuralODE(state_dim=state_dim, var_names=var_names, noise_scale=0.1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"✓ Successfully loaded {model_type}")
        return model
        
    except Exception as e:
        print(f"✗ Failed to load {model_type}: {e}")
        return None


def create_trained_model_variants(var_names, checkpoint_dir='checkpoints', device='cpu'):
    """Create model variants using trained checkpoints where available"""
    print("Loading trained models from checkpoints...")
    
    # Find available checkpoints
    checkpoint_map = find_checkpoint_files(checkpoint_dir)
    print(f"Found checkpoints: {list(checkpoint_map.keys())}")
    
    models = {}
    state_dim = len(var_names)
    
    # Define all model types we want to compare
    model_types = [
        'NODE_external',
        'NODE_external_ENSO',
        'NODE_internal', 
        'NODE_internal_ENSO',
        'PhysicsNODE_external',
        'PhysicsNODE_external_ENSO',
        'PhysicsNODE_internal',
        'PhysicsNODE_internal_ENSO'
    ]
    
    for model_type in model_types:
        if model_type in checkpoint_map:
            # Load trained model
            model = load_trained_model(
                checkpoint_map[model_type], 
                model_type, 
                var_names, 
                device
            )
            if model is not None:
                models[model_type] = model
        else:
            print(f"Warning: No checkpoint found for {model_type}, creating untrained model")
            # Create untrained model as fallback
            if model_type == 'NODE_external':
                models[model_type] = NeuralODE(state_dim=state_dim, var_names=var_names).to(device)
            elif model_type == 'NODE_external_ENSO':
                models[model_type] = NeuralODE(state_dim=state_dim, var_names=var_names).to(device)
            elif model_type == 'NODE_internal':
                models[model_type] = StochasticNeuralODE(state_dim=state_dim, var_names=var_names, noise_scale=0.1).to(device)
            elif model_type == 'NODE_internal_ENSO':
                models[model_type] = StochasticNeuralODE(state_dim=state_dim, var_names=var_names, noise_scale=0.1).to(device)
            elif model_type == 'PhysicsNODE_external':
                models[model_type] = PhysicsInformedODE(state_dim=state_dim, var_names=var_names).to(device)
            elif model_type == 'PhysicsNODE_external_ENSO':
                models[model_type] = PhysicsInformedODE(state_dim=state_dim, var_names=var_names).to(device)
            elif model_type == 'PhysicsNODE_internal':
                models[model_type] = StochasticNeuralODE(state_dim=state_dim, var_names=var_names, noise_scale=0.1).to(device)
            elif model_type == 'PhysicsNODE_internal_ENSO':
                models[model_type] = StochasticNeuralODE(state_dim=state_dim, var_names=var_names, noise_scale=0.1).to(device)
            
            models[model_type].eval()
    
    return models, checkpoint_map


def generate_forecasts_trained_models(models, obs_data, train_data, n_month=21, device='cpu'):
    """Generate forecasts for all models (trained and XRO)"""
    print("\nGenerating forecasts for all models...")
    
    forecasts = {}
    
    # 1. XRO forecasts (always trained/fitted)
    print("\n1. Generating XRO forecasts...")
    xro_model = XRO(ncycle=12, ac_order=2)
    xro_fit = xro_model.fit_matrix(train_data, maskb=['IOD'], maskNT=['T2', 'TH'])
    
    # Use same subset for XRO as neural models
    subset_data = obs_data.isel(time=slice(100, 400))
    
    xro_forecasts = xro_model.reforecast(
        fit_ds=xro_fit,
        init_ds=subset_data,
        n_month=n_month,
        ncopy=1,
        noise_type='zero'  # Deterministic for skill comparison
    )
    forecasts['XRO'] = xro_forecasts
    
    # 2-5. Neural ODE variants
    for i, (name, model) in enumerate(models.items(), 2):
        print(f"\n{i}. Generating {name} forecasts...")
        
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            try:
                # Use a subset of data for faster computation and to avoid memory issues
                subset_data = obs_data.isel(time=slice(100, 400))  # Use middle portion of data
                
                if 'internal' in name:
                    # Stochastic models - use deterministic mode for skill comparison
                    forecast = model.reforecast(
                        init_data=subset_data,
                        n_month=n_month,
                        ncopy=1,
                        enable_noise=False,  # Deterministic for fair comparison
                        device=device
                    )
                else:
                    # Regular models
                    forecast = model.reforecast(
                        init_data=subset_data,
                        n_month=n_month,
                        ncopy=1,
                        add_noise=False,  # Deterministic for fair comparison
                        device=device
                    )
                
                # Clean up the forecast coordinates for calc_forecast_skill
                forecast = forecast.drop_vars(['time', 'month'], errors='ignore')
                
                # Fix coordinate structure: the forecast has (init, lead) data but init coordinate
                # is indexed by time dimension. We need to properly set up the init coordinate.
                if 'init' in forecast.coords and 'lead' in forecast.coords:
                    # Rename init dimension and properly assign init coordinate
                    forecast = forecast.rename({'init': 'init_dim'})
                    forecast = forecast.assign_coords(init=('init_dim', subset_data.time.values))
                    forecast = forecast.swap_dims({'init_dim': 'init'})
                
                forecasts[name] = forecast
                print(f"✓ Successfully generated {name} forecasts")
                
            except Exception as e:
                print(f"✗ Failed to generate {name} forecasts: {e}")
    
    return forecasts


def compute_forecast_skills(forecasts, obs_data, verify_periods=slice('1979-01', '2022-12')):
    """Compute correlation and RMSE skills for all models"""
    print("\nComputing forecast skills...")
    
    skills = {
        'correlation': {},
        'rmse': {}
    }
    
    # Use subset of obs_data that matches forecast period
    obs_subset = obs_data.isel(time=slice(100, 400))
    
    for model_name, forecast in forecasts.items():
        print(f"Computing skills for {model_name}...")
        
        try:
            # Debug: print forecast structure
            print(f"  Forecast dims: {forecast.dims}")
            print(f"  Forecast coords: {list(forecast.coords.keys())}")
            
            # Correlation skill
            corr_skill = calc_forecast_skill(
                forecast, obs_subset,
                metric='pearson_r', is_mv3=False, by_month=False,  # Set is_mv3=False to avoid rolling mean issues
                verify_periods=slice('1987-01', '2020-12')  # Adjust to match subset period
            )
            skills['correlation'][model_name] = corr_skill
            
            # RMSE skill
            rmse_skill = calc_forecast_skill(
                forecast, obs_subset,
                metric='rmse', is_mv3=False, by_month=False,
                verify_periods=slice('1987-01', '2020-12')
            )
            skills['rmse'][model_name] = rmse_skill
            
            print(f"✓ Successfully computed skills for {model_name}")
            
        except Exception as e:
            print(f"✗ Failed to compute skills for {model_name}: {e}")
            # Create dummy skills with NaN values
            n_leads = len(forecast.lead) if 'lead' in forecast.dims else 21
            dummy_data = np.full(n_leads, np.nan)
            dummy_coords = {'lead': np.arange(n_leads)}
            
            skills['correlation'][model_name] = xr.Dataset({
                var: xr.DataArray(dummy_data, coords=dummy_coords, dims=['lead'])
                for var in obs_data.data_vars
            })
            skills['rmse'][model_name] = xr.Dataset({
                var: xr.DataArray(dummy_data, coords=dummy_coords, dims=['lead'])
                for var in obs_data.data_vars
            })
    
    return skills


def plot_forecast_skill_comparison(skills, variables=['Nino34'], save_path=None):
    """Plot correlation and RMSE skills for all models focusing on ENSO (Nino34)"""
    print("\nCreating ENSO forecast skill comparison plots...")
    
    # Focus only on ENSO (Nino34)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Single row, two columns
    
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
        'PhysicsNODE_internal_ENSO': {'color': 'blue', 'linestyle': '-.', 'linewidth': 2, 'marker': 'v', 'markersize': 4}
    }
    
    var = 'Nino34'  # Focus only on ENSO
    
    # Correlation skill plot
    ax_corr = axes[0]
    
    for model_name in skills['correlation'].keys():
        if var in skills['correlation'][model_name]:
            skill_data = skills['correlation'][model_name][var]
            
            if not skill_data.isnull().all():
                style = model_styles.get(model_name, {'color': 'gray', 'linestyle': '-'})
                skill_data.plot(
                    ax=ax_corr,
                    label=model_name,
                    **style
                )
    
    ax_corr.set_title(f'{var} - Correlation Skill')
    ax_corr.set_ylabel('Correlation')
    ax_corr.set_xlabel('Lead (months)')
    ax_corr.axhline(0.5, ls=':', color='gray', alpha=0.7, label='0.5 threshold')
    ax_corr.set_ylim([0, 1])
    ax_corr.grid(True, alpha=0.3)
    ax_corr.legend(fontsize=10, loc='upper right')
    
    # RMSE skill plot
    ax_rmse = axes[1]
    
    for model_name in skills['rmse'].keys():
        if var in skills['rmse'][model_name]:
            skill_data = skills['rmse'][model_name][var]
            
            if not skill_data.isnull().all():
                style = model_styles.get(model_name, {'color': 'gray', 'linestyle': '-'})
                skill_data.plot(
                    ax=ax_rmse,
                    label=model_name,
                    **style
                )
    
    ax_rmse.set_title(f'{var} - RMSE Skill')
    ax_rmse.set_ylabel('RMSE (°C)')
    ax_rmse.set_xlabel('Lead (months)')
    ax_rmse.grid(True, alpha=0.3)
    ax_rmse.legend(fontsize=10, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # plt.show()  # Commented out to avoid popup during training


def print_model_status(models, checkpoint_map):
    """Print status of loaded models"""
    print("\n" + "="*60)
    print("MODEL STATUS SUMMARY")
    print("="*60)
    
    all_model_types = ['NODE_external', 'NODE_external_ENSO', 'NODE_internal', 'NODE_internal_ENSO', 
                       'PhysicsNODE_external', 'PhysicsNODE_external_ENSO', 'PhysicsNODE_internal', 'PhysicsNODE_internal_ENSO']
    
    for model_type in all_model_types:
        if model_type in models:
            if model_type in checkpoint_map:
                status = f"✓ TRAINED (from {os.path.basename(checkpoint_map[model_type])})"
            else:
                status = "⚠ UNTRAINED (no checkpoint found)"
        else:
            status = "✗ NOT LOADED"
        
        print(f"{model_type:25} : {status}")


def main():
    """Main comprehensive comparison function"""
    print("="*80)
    print("TRAINED MODEL FORECAST SKILL COMPARISON")
    print("Models: XRO, NODE+external, NODE+external+ENSO, NODE+internal, NODE+internal+ENSO,")
    print("        PhysicsNODE+external, PhysicsNODE+external+ENSO, PhysicsNODE+internal, PhysicsNODE+internal+ENSO")
    print("="*80)
    
    # Load data
    print("Loading data...")
    obs_data = xr.open_dataset('data/XRO_indices_oras5.nc')
    var_names = list(obs_data.data_vars)
    
    # Split data
    train_data = obs_data.sel(time=slice('1979-01', '2022-12'))
    
    print(f"Variables: {var_names}")
    print(f"Training period: {train_data.time.min().values} to {train_data.time.max().values}")
    
    # Create trained model variants
    print("\nLoading trained models...")
    models, checkpoint_map = create_trained_model_variants(var_names, checkpoint_dir='checkpoints')
    
    # Print model status
    print_model_status(models, checkpoint_map)
    
    if not models:
        print("\n❌ No models loaded! Please train models first using:")
        print("  python train.py --model_type neural_ode --n_epochs 50")
        print("  python train.py --model_type neural_ode --enso_only --n_epochs 50")
        print("  python train.py --model_type physics_informed --n_epochs 50") 
        print("  python train.py --model_type physics_informed --enso_only --n_epochs 50")
        print("  python train.py --model_type stochastic --n_epochs 50")
        print("  python train.py --model_type stochastic --enso_only --n_epochs 50")
        return
    
    # Generate forecasts for all models
    forecasts = generate_forecasts_trained_models(
        models, obs_data, train_data, n_month=21, device='cpu'
    )
    
    print(f"\nGenerated forecasts for: {list(forecasts.keys())}")
    
    if len(forecasts) <= 1:  # Only XRO
        print("\n❌ No neural model forecasts generated! Check model loading.")
        return
    
    # Compute forecast skills
    skills = compute_forecast_skills(
        forecasts, obs_data, 
        verify_periods=slice('1979-01', '2022-12')
    )
    
    # Plot forecast skill comparisons (ENSO only)
    plot_forecast_skill_comparison(
        skills, 
        variables=['Nino34'],  # Focus only on ENSO
        save_path='trained_model_forecast_skills.png'
    )
    
    # Print summary
    print("\n" + "="*80)
    print("FORECAST SKILL SUMMARY")
    print("="*80)
    
    print("\nForecast Skills at 6-month lead (Nino34):")
    for model_name in skills['correlation'].keys():
        try:
            corr_6m = float(skills['correlation'][model_name]['Nino34'].isel(lead=5))
            rmse_6m = float(skills['rmse'][model_name]['Nino34'].isel(lead=5))
            print(f"  {model_name:25}: Corr = {corr_6m:.3f}, RMSE = {rmse_6m:.3f}")
        except:
            print(f"  {model_name:25}: Skills not available")
    
    print("\nForecast Skills at 12-month lead (Nino34):")
    for model_name in skills['correlation'].keys():
        try:
            corr_12m = float(skills['correlation'][model_name]['Nino34'].isel(lead=11))
            rmse_12m = float(skills['rmse'][model_name]['Nino34'].isel(lead=11))
            print(f"  {model_name:25}: Corr = {corr_12m:.3f}, RMSE = {rmse_12m:.3f}")
        except:
            print(f"  {model_name:25}: Skills not available")
    
    print(f"\nFiles created:")
    print(f"  - trained_model_forecast_skills.png")
    
    print(f"\nComparison Complete!")
    print(f"Compared {len(forecasts)} models including XRO baseline")


if __name__ == '__main__':
    main()
