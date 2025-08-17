"""
Demo training script for Neural ODE models
Quick example to show how to train and evaluate the models
"""

import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from model import PhysicsInformedODE
from train import create_data_loaders, Trainer
import os


def quick_demo():
    """Quick demonstration of the Neural ODE training and evaluation"""
    print("="*60)
    print("NEURAL ODE DEMO - QUICK TRAINING EXAMPLE")
    print("="*60)
    
    # Set device
    device = 'cpu'  # Use 'cuda' if available
    
    # Create data loaders
    print("1. Loading data...")
    train_loader, val_loader, var_names = create_data_loaders(
        'data/XRO_indices_oras5.nc',
        train_split=0.8,
        sequence_length=12,  # Shorter for demo
        forecast_horizon=6,  # Shorter for demo
        batch_size=8        # Smaller batch for demo
    )
    
    print(f"   Variables: {var_names}")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    # Create model
    print("\n2. Creating Physics-Informed Neural ODE model...")
    model = PhysicsInformedODE(
        state_dim=len(var_names),
        ncycle=12,
        ac_order=2,
        hidden_dim=32,  # Smaller for demo
        var_names=var_names
    )
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create trainer
    print("\n3. Setting up trainer...")
    trainer = Trainer(model, train_loader, val_loader, device)
    
    # Quick training (just a few epochs for demo)
    print("\n4. Training model (demo: 5 epochs)...")
    n_epochs = 5
    save_path = 'checkpoints/demo_physics_informed.pt'
    
    history = trainer.train(n_epochs, save_path)
    
    # Plot training curves
    print("\n5. Plotting training curves...")
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], 'b-', label='Train', linewidth=2)
    plt.plot(history['val_losses'], 'r-', label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Curves')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(history['train_losses'], 'b-', label='Train', linewidth=2)
    plt.semilogy(history['val_losses'], 'r-', label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.title('Training Curves (Log Scale)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Test simulation
    print("\n6. Testing simulation...")
    data = xr.open_dataset('data/XRO_indices_oras5.nc')
    x0_data = data.isel(time=0)
    
    model.eval()
    with torch.no_grad():
        sim_result = model.simulate(
            x0_data=x0_data,
            nyear=2,
            ncopy=5,
            add_noise=True,
            device=device
        )
    
    # Plot simulation results
    print("\n7. Plotting simulation results...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    variables_to_plot = ['Nino34', 'WWV', 'IOD', 'NPMM']
    
    for i, var in enumerate(variables_to_plot):
        ax = axes[i]
        
        # Plot ensemble members
        for member in range(min(3, sim_result.member.size)):  # Plot first 3 members
            sim_result[var].isel(member=member).plot(
                ax=ax, alpha=0.7, linewidth=1, 
                label=f'Member {member+1}' if i == 0 else ""
            )
        
        # Plot ensemble mean
        sim_result[var].mean('member').plot(
            ax=ax, color='black', linewidth=2,
            label='Ensemble Mean' if i == 0 else ""
        )
        
        ax.set_title(f'{var} Simulation')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('demo_simulation_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Test forecasting
    print("\n8. Testing forecasting...")
    init_data = data.isel(time=slice(100, 110))  # 10 initialization times
    
    with torch.no_grad():
        forecast_result = model.reforecast(
            init_data=init_data,
            n_month=12,
            ncopy=3,
            add_noise=False,
            device=device
        )
    
    # Plot forecast example
    print("\n9. Plotting forecast example...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    init_idx = 5  # Choose one initialization
    
    for i, var in enumerate(variables_to_plot):
        ax = axes[i]
        
        # Plot forecast
        forecast_result[var].isel(init=init_idx).plot(
            ax=ax, color='red', linewidth=2, label='Forecast'
        )
        
        # Plot verification (if available)
        init_time = init_data.time[init_idx]
        verification = data[var].sel(time=slice(init_time, None)).isel(time=slice(0, 13))
        
        if len(verification) > 0:
            verification.plot(
                ax=ax, color='black', linewidth=2, 
                alpha=0.7, label='Observation'
            )
        
        ax.set_title(f'{var} Forecast')
        ax.set_xlabel('Lead (months)')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('demo_forecast_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY! ✓")
    print("="*60)
    
    print(f"\nFiles created:")
    print(f"- Model checkpoint: {save_path}")
    print(f"- Training curves: demo_training_curves.png")
    print(f"- Simulation results: demo_simulation_results.png")
    print(f"- Forecast results: demo_forecast_results.png")
    
    print(f"\nNext steps:")
    print(f"1. For longer training: python train.py --model_type physics_informed --n_epochs 100")
    print(f"2. For evaluation: python evaluate_neural_ode.py --checkpoint {save_path} --model_type physics_informed")
    print(f"3. Compare with XRO using the evaluation script")


if __name__ == '__main__':
    quick_demo()
