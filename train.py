"""
Training script for Neural ODE climate models
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import argparse
import json
import os
from datetime import datetime

from model import NeuralODE, PhysicsInformedODE, StochasticNeuralODE, OscillatorNeuralODE, GraphNeuralODE, PhysicsGraphNeuralODE, SineGraphNeuralODE, SinePhysicsGraphNeuralODE
from utils import calc_forecast_skill


class ClimateDataset(Dataset):
    """
    Dataset class for climate time series data
    """
    def __init__(self, data: xr.Dataset, sequence_length: int = 24, 
                 forecast_horizon: int = 12, var_names: List[str] = None):
        """
        Args:
            data: xarray Dataset with climate variables
            sequence_length: Length of input sequences (months)
            forecast_horizon: Length of forecast horizon (months)
            var_names: List of variable names to use
        """
        self.data = data
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.var_names = var_names or list(data.data_vars)
        
        # Convert to numpy arrays
        self.time_series = np.stack([data[var].values for var in self.var_names], axis=0)
        self.n_vars, self.n_times = self.time_series.shape
        
        # Normalize data
        self.mean = np.mean(self.time_series, axis=1, keepdims=True)
        self.std = np.std(self.time_series, axis=1, keepdims=True)
        self.normalized_data = (self.time_series - self.mean) / (self.std + 1e-8)
        
        # Create valid indices for sequences
        self.valid_indices = list(range(
            self.sequence_length, 
            self.n_times - self.forecast_horizon
        ))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        Returns:
            input_seq: [n_vars, sequence_length]
            target_seq: [n_vars, forecast_horizon]
            time_points: [sequence_length + forecast_horizon]
        """
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.sequence_length
        
        # Input sequence
        input_seq = self.normalized_data[:, start_idx:end_idx]
        
        # Target sequence (forecast)
        target_seq = self.normalized_data[:, end_idx:end_idx + self.forecast_horizon]
        
        # Time points (normalized to [0, 1] for one year cycles)
        total_length = self.sequence_length + self.forecast_horizon
        time_points = np.linspace(0, total_length / 12.0, total_length)
        
        return (
            torch.tensor(input_seq.T, dtype=torch.float32),  # [seq_len, n_vars]
            torch.tensor(target_seq.T, dtype=torch.float32),  # [forecast_horizon, n_vars]
            torch.tensor(time_points, dtype=torch.float32)
        )
    
    def denormalize(self, normalized_data: np.ndarray) -> np.ndarray:
        """Denormalize data back to original scale"""
        return normalized_data * self.std + self.mean


class ForecastLoss(nn.Module):
    """
    Custom loss function for forecasting that weights different lead times
    """
    def __init__(self, lead_weights: Optional[torch.Tensor] = None, enso_only: bool = False, var_names: Optional[List[str]] = None):
        super().__init__()
        self.lead_weights = lead_weights
        self.enso_only = enso_only
        self.var_names = var_names
        self.mse = nn.MSELoss(reduction='none')
        
        # Find ENSO index (Nino34) if enso_only is True
        if self.enso_only and self.var_names:
            try:
                self.enso_idx = self.var_names.index('Nino34')
                print(f"ENSO-only loss: Using variable '{self.var_names[self.enso_idx]}' at index {self.enso_idx}")
            except ValueError:
                print("Warning: 'Nino34' not found in var_names, using first variable as ENSO proxy")
                self.enso_idx = 0
        elif self.enso_only:
            print("Warning: enso_only=True but no var_names provided, using first variable")
            self.enso_idx = 0
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size, forecast_horizon, n_vars]
            targets: [batch_size, forecast_horizon, n_vars]
        Returns:
            Weighted MSE loss (ENSO-only if specified)
        """
        if self.enso_only:
            # Compute loss only for ENSO variable
            mse_loss = self.mse(predictions[:, :, self.enso_idx:self.enso_idx+1], 
                               targets[:, :, self.enso_idx:self.enso_idx+1])  # [batch_size, forecast_horizon, 1]
        else:
            # Compute loss for all variables
            mse_loss = self.mse(predictions, targets)  # [batch_size, forecast_horizon, n_vars]
        
        if self.lead_weights is not None:
            # Apply lead time weights
            if self.enso_only:
                weights = self.lead_weights.view(1, -1, 1)  # [1, forecast_horizon, 1]
            else:
                weights = self.lead_weights.view(1, -1, 1)  # [1, forecast_horizon, 1]
            mse_loss = mse_loss * weights
        
        return mse_loss.mean()


class Trainer:
    """
    Trainer class for Neural ODE models
    """
    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 val_loader: DataLoader, device: str = 'cpu', enso_only: bool = False):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.enso_only = enso_only
        
        # Get variable names from dataset
        var_names = getattr(train_loader.dataset, 'var_names', None)
        
        # Loss function with lead time weighting (emphasize shorter leads)
        lead_weights = torch.exp(-0.1 * torch.arange(train_loader.dataset.forecast_horizon))
        self.criterion = ForecastLoss(lead_weights.to(device), enso_only=enso_only, var_names=var_names)
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch_idx, (input_seq, target_seq, time_points) in enumerate(self.train_loader):
            input_seq = input_seq.to(self.device)  # [batch_size, seq_len, n_vars]
            target_seq = target_seq.to(self.device)  # [batch_size, forecast_horizon, n_vars]
            time_points = time_points.to(self.device)  # [batch_size, total_time]
            
            batch_size, seq_len, n_vars = input_seq.shape
            forecast_horizon = target_seq.shape[1]
            
            # Use last state as initial condition
            x0 = input_seq[:, -1, :]  # [batch_size, n_vars]
            
            # Time points for forecast
            t_forecast = time_points[:, seq_len:]  # [batch_size, forecast_horizon]
            
            self.optimizer.zero_grad()
            
            try:
                # Forward pass - predict future states
                predictions = []
                for i in range(batch_size):
                    # Integrate from initial condition
                    if hasattr(self.model, 'forward') and 'enable_noise' in self.model.forward.__code__.co_varnames:
                        # Stochastic model
                        pred = self.model(
                            x0[i:i+1], 
                            t_forecast[i], 
                            enable_noise=False  # Disable noise during training for stability
                        )  # [1, forecast_horizon, n_vars]
                    else:
                        # Regular models
                        pred = self.model(
                            x0[i:i+1], 
                            t_forecast[i], 
                            add_noise=False
                        )  # [1, forecast_horizon, n_vars]
                    predictions.append(pred)
                
                predictions = torch.cat(predictions, dim=0)  # [batch_size, forecast_horizon, n_vars]
                
                # Compute loss
                loss = self.criterion(predictions, target_seq)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.6f}')
                    
            except Exception as e:
                print(f'Error in batch {batch_idx}: {e}')
                continue
        
        return total_loss / max(n_batches, 1)
    
    def validate(self) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for input_seq, target_seq, time_points in self.val_loader:
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                time_points = time_points.to(self.device)
                
                batch_size, seq_len, n_vars = input_seq.shape
                
                # Use last state as initial condition
                x0 = input_seq[:, -1, :]
                
                # Time points for forecast
                t_forecast = time_points[:, seq_len:]
                
                try:
                    # Forward pass
                    predictions = []
                    for i in range(batch_size):
                        if hasattr(self.model, 'forward') and 'enable_noise' in self.model.forward.__code__.co_varnames:
                            # Stochastic model
                            pred = self.model(
                                x0[i:i+1], 
                                t_forecast[i], 
                                enable_noise=False
                            )
                        else:
                            # Regular models
                            pred = self.model(
                                x0[i:i+1], 
                                t_forecast[i], 
                                add_noise=False
                            )
                        predictions.append(pred)
                    
                    predictions = torch.cat(predictions, dim=0)
                    
                    # Compute loss
                    loss = self.criterion(predictions, target_seq)
                    total_loss += loss.item()
                    n_batches += 1
                    
                except Exception as e:
                    print(f'Validation error: {e}')
                    continue
        
        return total_loss / max(n_batches, 1)
    
    def train(self, n_epochs: int, save_path: str = None, model_type: str = None, 
              save_dir: str = None, timestamp: str = None) -> Dict:
        """
        Train the model
        
        Args:
            n_epochs: Number of epochs to train
            save_path: Path to save the best model
            model_type: Type of model being trained (for trajectory plotting)
            save_dir: Directory to save additional outputs
            timestamp: Timestamp for file naming
        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        
        for epoch in range(n_epochs):
            print(f'\nEpoch {epoch+1}/{n_epochs}')
            print('-' * 50)
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            print(f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # Save best model
            if val_loss < best_val_loss and save_path:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, save_path)
                print(f'Saved best model with val_loss: {val_loss:.6f}')
                
                # Generate forecast trajectory plot for best model
                if model_type and save_dir and timestamp:
                    self.plot_forecast_trajectory(save_dir, model_type, timestamp)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def plot_forecast_trajectory(self, save_dir: str, model_type: str, timestamp: str):
        """
        Generate and save forecast trajectory vs target trajectory plot for ENSO
        """
        try:
            print("Generating forecast trajectory plot...")
            
            # Use a few validation samples for trajectory plotting
            self.model.eval()
            with torch.no_grad():
                # Get a batch from validation loader
                for input_seq, target_seq, time_points in self.val_loader:
                    input_seq = input_seq.to(self.device)
                    target_seq = target_seq.to(self.device)
                    time_points = time_points.to(self.device)
                    
                    batch_size, seq_len, n_vars = input_seq.shape
                    forecast_horizon = target_seq.shape[1]
                    
                    # Use last state as initial condition
                    x0 = input_seq[:, -1, :]
                    t_forecast = time_points[:, seq_len:]
                    
                    # Generate predictions for first few samples
                    n_samples = min(3, batch_size)
                    predictions = []
                    
                    for i in range(n_samples):
                        if hasattr(self.model, 'forward') and 'enable_noise' in self.model.forward.__code__.co_varnames:
                            pred = self.model(x0[i:i+1], t_forecast[i], enable_noise=False)
                        else:
                            pred = self.model(x0[i:i+1], t_forecast[i], add_noise=False)
                        predictions.append(pred)
                    
                    predictions = torch.cat(predictions, dim=0)  # [n_samples, forecast_horizon, n_vars]
                    
                    # Plot trajectories
                    self._create_trajectory_plot(
                        predictions[:n_samples], 
                        target_seq[:n_samples], 
                        input_seq[:n_samples], 
                        save_dir, 
                        model_type, 
                        timestamp
                    )
                    break  # Only use first batch
                    
        except Exception as e:
            print(f"Warning: Could not generate trajectory plot: {e}")
    
    def _create_trajectory_plot(self, predictions, targets, inputs, save_dir, model_type, timestamp):
        """Create the actual trajectory plot"""
        import matplotlib.pyplot as plt
        
        # Convert to numpy
        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        inputs_np = inputs.cpu().numpy()
        
        # Find ENSO index (Nino34)
        var_names = getattr(self.train_loader.dataset, 'var_names', None)
        enso_idx = 0  # Default to first variable
        if var_names and 'Nino34' in var_names:
            enso_idx = var_names.index('Nino34')
        
        n_samples, seq_len, _ = inputs_np.shape
        forecast_horizon = predictions_np.shape[1]
        
        # Create plot
        fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4*n_samples))
        if n_samples == 1:
            axes = [axes]
        
        for i in range(n_samples):
            ax = axes[i]
            
            # Time axes
            input_time = np.arange(-seq_len, 0)
            forecast_time = np.arange(0, forecast_horizon)
            
            # Plot input sequence (context)
            ax.plot(input_time, inputs_np[i, :, enso_idx], 'k-', linewidth=2, 
                   label='Input Context', alpha=0.7)
            
            # Plot target trajectory (truth)
            ax.plot(forecast_time, targets_np[i, :, enso_idx], 'b-', linewidth=3,
                   label='True Target', marker='o', markersize=4)
            
            # Plot predicted trajectory
            ax.plot(forecast_time, predictions_np[i, :, enso_idx], 'r--', linewidth=3,
                   label='Model Forecast', marker='s', markersize=4)
            
            # Formatting
            ax.axvline(0, color='gray', linestyle=':', alpha=0.5, label='Forecast Start')
            ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
            ax.axhline(0.5, color='red', linestyle='--', alpha=0.3)
            ax.axhline(-0.5, color='blue', linestyle='--', alpha=0.3)
            
            ax.set_xlabel('Time (months)')
            ax.set_ylabel('ENSO (Nino3.4) Anomaly')
            ax.set_title(f'Sample {i+1}: Forecast vs Target Trajectory')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_type} - ENSO Forecast Trajectory Comparison', fontsize=14, y=0.98)
        plt.tight_layout()
        
        # Save plot
        model_suffix = '_enso_only' if self.enso_only else ''
        plot_path = os.path.join(save_dir, f'{model_type}{model_suffix}_{timestamp}_trajectory.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved trajectory plot: {plot_path}")


def create_data_loaders(data_path: str, train_split: float = 0.8, 
                       sequence_length: int = 24, forecast_horizon: int = 12,
                       batch_size: int = 32) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Create training and validation data loaders
    
    Args:
        data_path: Path to the NetCDF data file
        train_split: Fraction of data to use for training
        sequence_length: Length of input sequences
        forecast_horizon: Length of forecast horizon
        batch_size: Batch size
    Returns:
        train_loader, val_loader, variable_names
    """
    # Load data
    data = xr.open_dataset(data_path)
    var_names = list(data.data_vars)
    
    # Split data temporally
    n_times = len(data.time)
    split_idx = int(n_times * train_split)
    
    train_data = data.isel(time=slice(0, split_idx))
    val_data = data.isel(time=slice(split_idx, None))
    
    # Create datasets
    train_dataset = ClimateDataset(train_data, sequence_length, forecast_horizon, var_names)
    val_dataset = ClimateDataset(val_data, sequence_length, forecast_horizon, var_names)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, val_loader, var_names


def main():
    parser = argparse.ArgumentParser(description='Train Neural ODE for climate forecasting')
    parser.add_argument('--data_path', type=str, default='data/XRO_indices_oras5.nc',
                       help='Path to the data file')
    parser.add_argument('--model_type', type=str, 
                       choices=['neural_ode', 'physics_informed', 'stochastic', 
                               'oscillator_external', 'oscillator_internal',
                               'graph_gcn', 'graph_gat', 'physics_graph_gcn', 'physics_graph_gat',
                               'sine_graph_sine', 'sine_graph_kuramoto', 'sine_physics_graph_sine', 'sine_physics_graph_kuramoto'], 
                       default='physics_informed', help='Type of model to train')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--sequence_length', type=int, default=24, help='Input sequence length (months)')
    parser.add_argument('--forecast_horizon', type=int, default=12, help='Forecast horizon (months)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save models')
    parser.add_argument('--enso_only', action='store_true', help='Train using loss only from ENSO (Nino34) variable')
    
    args = parser.parse_args()
    
    # Auto-detect device if not specified
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = 'cuda'
            print(f'CUDA is available - using GPU: {torch.cuda.get_device_name(0)}')
        else:
            args.device = 'cpu'
            print('CUDA not available - using CPU')
    else:
        print(f'Using device: {args.device}')
    
    # Create save directory and model-specific subdirectory
    os.makedirs(args.save_dir, exist_ok=True)
    model_suffix = '_enso_only' if args.enso_only else ''
    model_save_dir = os.path.join(args.save_dir, f'{args.model_type}{model_suffix}')
    os.makedirs(model_save_dir, exist_ok=True)
    print(f'Created model directory: {model_save_dir}')
    
    # Create data loaders
    print('Loading data...')
    train_loader, val_loader, var_names = create_data_loaders(
        args.data_path, 
        sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon,
        batch_size=args.batch_size
    )
    
    print(f'Variables: {var_names}')
    print(f'Training batches: {len(train_loader)}')
    print(f'Validation batches: {len(val_loader)}')
    
    # Create model
    state_dim = len(var_names)
    if args.model_type == 'neural_ode':
        model = NeuralODE(
            state_dim=state_dim,
            hidden_dim=args.hidden_dim,
            var_names=var_names
        )
    elif args.model_type == 'stochastic':
        model = StochasticNeuralODE(
            state_dim=state_dim,
            hidden_dim=args.hidden_dim,
            var_names=var_names,
            noise_scale=0.1  # Start with small noise
        )
    elif args.model_type == 'oscillator_external':
        model = OscillatorNeuralODE(
            state_dim=state_dim,
            hidden_dim=args.hidden_dim,
            var_names=var_names,
            noise_mode='external'
        )
    elif args.model_type == 'oscillator_internal':
        model = OscillatorNeuralODE(
            state_dim=state_dim,
            hidden_dim=args.hidden_dim,
            var_names=var_names,
            noise_mode='internal',
            noise_scale=0.1
        )
    elif args.model_type == 'graph_gcn':
        model = GraphNeuralODE(
            state_dim=state_dim,
            var_names=var_names,
            hidden_dim=args.hidden_dim,
            gnn_type='gcn',
            noise_mode='external'
        )
    elif args.model_type == 'graph_gat':
        model = GraphNeuralODE(
            state_dim=state_dim,
            var_names=var_names,
            hidden_dim=args.hidden_dim,
            gnn_type='gat',
            noise_mode='external'
        )
    elif args.model_type == 'physics_graph_gcn':
        model = PhysicsGraphNeuralODE(
            state_dim=state_dim,
            var_names=var_names,
            hidden_dim=args.hidden_dim,
            gnn_type='gcn',
            noise_mode='external'
        )
    elif args.model_type == 'physics_graph_gat':
        model = PhysicsGraphNeuralODE(
            state_dim=state_dim,
            var_names=var_names,
            hidden_dim=args.hidden_dim,
            gnn_type='gat',
            noise_mode='external'
        )
    elif args.model_type == 'sine_graph_sine':
        model = SineGraphNeuralODE(
            state_dim=state_dim,
            var_names=var_names,
            hidden_dim=args.hidden_dim,
            sine_type='sine',
            noise_mode='external'
        )
    elif args.model_type == 'sine_graph_kuramoto':
        model = SineGraphNeuralODE(
            state_dim=state_dim,
            var_names=var_names,
            hidden_dim=args.hidden_dim,
            sine_type='kuramoto',
            noise_mode='external'
        )
    elif args.model_type == 'sine_physics_graph_sine':
        model = SinePhysicsGraphNeuralODE(
            state_dim=state_dim,
            var_names=var_names,
            hidden_dim=args.hidden_dim,
            sine_type='sine',
            noise_mode='external'
        )
    elif args.model_type == 'sine_physics_graph_kuramoto':
        model = SinePhysicsGraphNeuralODE(
            state_dim=state_dim,
            var_names=var_names,
            hidden_dim=args.hidden_dim,
            sine_type='kuramoto',
            noise_mode='external'
        )
    else:  # physics_informed
        model = PhysicsInformedODE(
            state_dim=state_dim,
            hidden_dim=args.hidden_dim,
            var_names=var_names
        )
    
    print(f'Model: {args.model_type}')
    print(f'Parameters: {sum(p.numel() for p in model.parameters())}')
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, args.device, enso_only=getattr(args, 'enso_only', False))
    
    # Train
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(model_save_dir, f'{args.model_type}{model_suffix}_{timestamp}.pt')
    
    print('Starting training...')
    history = trainer.train(args.n_epochs, save_path, args.model_type, model_save_dir, timestamp)
    
    # Save training history
    history_path = os.path.join(model_save_dir, f'{args.model_type}{model_suffix}_{timestamp}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)
    
    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train')
    plt.plot(history['val_losses'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Curves')
    
    plt.subplot(1, 2, 2)
    plt.semilogy(history['train_losses'], label='Train')
    plt.semilogy(history['val_losses'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.title('Training Curves (Log Scale)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, f'{args.model_type}{model_suffix}_{timestamp}_curves.png'))
    # plt.show()  # Commented out to avoid popup during training
    
    print(f'Training completed. Best model saved to: {save_path}')
    print(f'All outputs saved in: {model_save_dir}')


if __name__ == '__main__':
    main()
