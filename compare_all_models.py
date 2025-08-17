"""
Comprehensive comparison of all Neural ODE models vs XRO
"""

import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from model import NeuralODE, PhysicsInformedODE, StochasticNeuralODE
from xro.XRO import XRO
import time


def create_untrained_models(var_names, device='cpu'):
    """Create untrained models for comparison"""
    state_dim = len(var_names)
    
    models = {
        'NeuralODE': NeuralODE(state_dim=state_dim, var_names=var_names),
        'PhysicsInformed': PhysicsInformedODE(state_dim=state_dim, var_names=var_names),
        'Stochastic': StochasticNeuralODE(state_dim=state_dim, var_names=var_names, noise_scale=0.1)
    }
    
    # Move to device
    for name, model in models.items():
        models[name] = model.to(device)
        models[name].eval()
    
    return models


def compare_simulation_speed(models, x0_data, nyear=2, ncopy=5):
    """Compare simulation speed of different models"""
    print("Comparing simulation speed...")
    
    timing_results = {}
    
    for name, model in models.items():
        print(f"\nTesting {name}...")
        
        # Warm up
        with torch.no_grad():
            if name == 'Stochastic':
                _ = model.simulate(x0_data, nyear=1, ncopy=1, enable_noise=False)
            else:
                _ = model.simulate(x0_data, nyear=1, ncopy=1, add_noise=False)
        
        # Time the simulation
        start_time = time.time()
        
        with torch.no_grad():
            if name == 'Stochastic':
                sim = model.simulate(x0_data, nyear=nyear, ncopy=ncopy, enable_noise=True)
            else:
                sim = model.simulate(x0_data, nyear=nyear, ncopy=ncopy, add_noise=True)
        
        end_time = time.time()
        
        timing_results[name] = {
            'time': end_time - start_time,
            'simulation': sim
        }
        
        print(f"{name} simulation time: {end_time - start_time:.3f} seconds")
    
    return timing_results


def compare_with_xro_baseline(models, train_data, x0_data, nyear=2, ncopy=5):
    """Compare with XRO baseline"""
    print("\nComparing with XRO baseline...")
    
    # Create and fit XRO model
    xro_model = XRO(ncycle=12, ac_order=2)
    xro_fit = xro_model.fit_matrix(train_data, maskb=['IOD'], maskNT=['T2', 'TH'])
    
    # XRO simulation
    start_time = time.time()
    xro_sim = xro_model.simulate(
        fit_ds=xro_fit,
        X0_ds=x0_data,
        nyear=nyear,
        ncopy=ncopy,
        is_xi_stdac=False,
        seed=42
    )
    xro_time = time.time() - start_time
    
    print(f"XRO simulation time: {xro_time:.3f} seconds")
    
    return xro_sim, xro_time


def analyze_ensemble_spread(simulations, var_name='Nino34'):
    """Analyze ensemble spread characteristics"""
    print(f"\nAnalyzing ensemble spread for {var_name}...")
    
    spread_stats = {}
    
    for name, sim_data in simulations.items():
        if 'simulation' in sim_data:
            sim = sim_data['simulation']
        else:
            sim = sim_data  # For XRO
        
        if var_name in sim and 'member' in sim.dims:
            var_data = sim[var_name]
            
            # Calculate ensemble statistics
            ens_mean = var_data.mean('member')
            ens_std = var_data.std('member')
            ens_range = var_data.max('member') - var_data.min('member')
            
            spread_stats[name] = {
                'mean_spread': float(ens_std.mean()),
                'max_spread': float(ens_std.max()),
                'mean_range': float(ens_range.mean()),
                'time_series': {
                    'mean': ens_mean,
                    'std': ens_std,
                    'range': ens_range
                }
            }
            
            print(f"{name}: Mean spread = {spread_stats[name]['mean_spread']:.3f}, "
                  f"Max spread = {spread_stats[name]['max_spread']:.3f}")
    
    return spread_stats


def plot_comprehensive_comparison(neural_simulations, xro_sim, spread_stats, var_name='Nino34'):
    """Create comprehensive comparison plots"""
    print(f"\nCreating comprehensive comparison plots for {var_name}...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Time series comparison
    ax1 = plt.subplot(3, 2, 1)
    colors = {'NeuralODE': 'red', 'PhysicsInformed': 'blue', 'Stochastic': 'green'}
    
    for name, sim_data in neural_simulations.items():
        sim = sim_data['simulation']
        if var_name in sim:
            # Plot ensemble mean
            sim[var_name].mean('member').plot(ax=ax1, color=colors[name], 
                                            linewidth=2, label=f'{name} (Mean)')
            
            # Plot ensemble spread
            ens_mean = sim[var_name].mean('member')
            ens_std = sim[var_name].std('member')
            ax1.fill_between(ens_mean.time, ens_mean - ens_std, ens_mean + ens_std,
                           color=colors[name], alpha=0.2)
    
    # Add XRO
    if var_name in xro_sim:
        xro_mean = xro_sim[var_name].mean('member')
        xro_std = xro_sim[var_name].std('member')
        xro_mean.plot(ax=ax1, color='black', linewidth=2, label='XRO (Mean)')
        ax1.fill_between(xro_mean.time, xro_mean - xro_std, xro_mean + xro_std,
                        color='black', alpha=0.2)
    
    ax1.set_title(f'{var_name} - Ensemble Simulations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Ensemble spread comparison
    ax2 = plt.subplot(3, 2, 2)
    spread_names = list(spread_stats.keys())
    spread_values = [spread_stats[name]['mean_spread'] for name in spread_names]
    
    bars = ax2.bar(spread_names, spread_values, color=['red', 'blue', 'green', 'black'])
    ax2.set_title(f'{var_name} - Mean Ensemble Spread')
    ax2.set_ylabel('Standard Deviation')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, spread_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 3. Individual ensemble members (first model)
    ax3 = plt.subplot(3, 2, 3)
    first_model_name = list(neural_simulations.keys())[0]
    first_sim = neural_simulations[first_model_name]['simulation']
    
    if var_name in first_sim:
        for i in range(min(5, first_sim.member.size)):
            first_sim[var_name].isel(member=i).plot(ax=ax3, alpha=0.7, linewidth=1)
        
        first_sim[var_name].mean('member').plot(ax=ax3, color='black', linewidth=2, label='Ensemble Mean')
    
    ax3.set_title(f'{var_name} - {first_model_name} Individual Members')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Spread evolution over time
    ax4 = plt.subplot(3, 2, 4)
    for name in spread_stats.keys():
        if 'time_series' in spread_stats[name]:
            spread_stats[name]['time_series']['std'].plot(ax=ax4, label=name, linewidth=2)
    
    ax4.set_title(f'{var_name} - Ensemble Spread Evolution')
    ax4.set_ylabel('Standard Deviation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Distribution comparison (final time step)
    ax5 = plt.subplot(3, 2, 5)
    final_values = {}
    
    for name, sim_data in neural_simulations.items():
        sim = sim_data['simulation']
        if var_name in sim:
            final_vals = sim[var_name].isel(time=-1).values
            ax5.hist(final_vals, alpha=0.6, label=name, bins=10)
            final_values[name] = final_vals
    
    # Add XRO
    if var_name in xro_sim:
        xro_final = xro_sim[var_name].isel(time=-1).values
        ax5.hist(xro_final, alpha=0.6, label='XRO', bins=10, color='black')
        final_values['XRO'] = xro_final
    
    ax5.set_title(f'{var_name} - Final Value Distribution')
    ax5.set_xlabel('Value')
    ax5.set_ylabel('Frequency')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Performance summary table
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    
    # Create summary table
    table_data = []
    headers = ['Model', 'Mean Spread', 'Max Spread', 'Final Mean', 'Final Std']
    
    for name in spread_stats.keys():
        if name in final_values:
            final_mean = np.mean(final_values[name])
            final_std = np.std(final_values[name])
            
            row = [
                name,
                f"{spread_stats[name]['mean_spread']:.3f}",
                f"{spread_stats[name]['max_spread']:.3f}",
                f"{final_mean:.3f}",
                f"{final_std:.3f}"
            ]
            table_data.append(row)
    
    table = ax6.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax6.set_title('Performance Summary', pad=20)
    
    plt.tight_layout()
    plt.savefig('comprehensive_model_comparison.png', dpi=150, bbox_inches='tight')
    # plt.show()  # Commented out to avoid popup during training


def main():
    """Main comparison function"""
    print("="*60)
    print("COMPREHENSIVE NEURAL ODE MODEL COMPARISON")
    print("="*60)
    
    # Load data
    print("Loading data...")
    data = xr.open_dataset('data/XRO_indices_oras5.nc')
    var_names = list(data.data_vars)
    
    # Split data
    train_data = data.sel(time=slice('1979-01', '2022-12'))
    x0_data = data.isel(time=0)
    
    print(f"Variables: {var_names}")
    print(f"Training period: {train_data.time.min().values} to {train_data.time.max().values}")
    
    # Create models
    print("\nCreating untrained models...")
    neural_models = create_untrained_models(var_names)
    
    print(f"Created models: {list(neural_models.keys())}")
    
    # Compare simulation speed
    neural_timing = compare_simulation_speed(neural_models, x0_data, nyear=2, ncopy=5)
    
    # Compare with XRO
    xro_sim, xro_time = compare_with_xro_baseline(neural_models, train_data, x0_data, nyear=2, ncopy=5)
    
    # Combine all simulations for analysis
    all_simulations = neural_timing.copy()
    all_simulations['XRO'] = xro_sim
    
    # Analyze ensemble spread
    spread_stats = analyze_ensemble_spread(all_simulations, 'Nino34')
    
    # Create comprehensive plots
    plot_comprehensive_comparison(neural_timing, xro_sim, spread_stats, 'Nino34')
    
    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print("\nTiming Results:")
    for name, data in neural_timing.items():
        print(f"  {name}: {data['time']:.3f} seconds")
    print(f"  XRO: {xro_time:.3f} seconds")
    
    print(f"\nEnsemble Spread (Nino34):")
    for name, stats in spread_stats.items():
        print(f"  {name}: Mean = {stats['mean_spread']:.3f}, Max = {stats['max_spread']:.3f}")
    
    print(f"\nFiles created:")
    print(f"  - comprehensive_model_comparison.png")
    
    print(f"\nNext steps:")
    print(f"1. Train models: python train.py --model_type [neural_ode|physics_informed|stochastic]")
    print(f"2. Evaluate trained models with real performance metrics")
    print(f"3. Compare forecast skill against observations")


if __name__ == '__main__':
    main()
