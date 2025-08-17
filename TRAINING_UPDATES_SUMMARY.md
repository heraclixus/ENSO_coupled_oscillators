# Training Updates Summary

## Overview
This document summarizes the updates made to the model training system to implement organized checkpoint storage and ENSO forecast trajectory visualization.

## Key Changes Implemented

### 1. Model-Specific Subdirectories
- **Location**: `train.py` main function
- **Change**: Each model now creates its own subdirectory under `checkpoints/`
- **Structure**: 
  ```
  checkpoints/
  ├── physics_informed/
  │   ├── physics_informed_20250101_120000.pt
  │   ├── physics_informed_20250101_120000_history.json
  │   ├── physics_informed_20250101_120000_curves.png
  │   └── physics_informed_20250101_120000_trajectory.png
  ├── physics_informed_enso_only/
  │   └── [similar files for ENSO-only training]
  ├── neural_ode/
  ├── stochastic/
  └── [other model types...]
  ```

### 2. ENSO Forecast Trajectory Plotting
- **Location**: `train.py` Trainer class
- **New Methods**:
  - `plot_forecast_trajectory()`: Generates trajectory plots during training
  - `_create_trajectory_plot()`: Creates the actual visualization
- **Features**:
  - Shows input context, true target, and model forecast
  - Focuses on ENSO (Nino3.4) variable
  - Generated automatically when best model is saved
  - Saves as `{model_type}_{timestamp}_trajectory.png`

### 3. Enhanced Trainer Class
- **Location**: `train.py` Trainer.train() method
- **Updates**:
  - Added parameters: `model_type`, `save_dir`, `timestamp`
  - Automatic trajectory plot generation for best models
  - Better integration with model-specific directories

### 4. Updated Comparison Scripts
- **Files**: `trained_model_comparison.py`, `trained_model_comparison_advanced.py`
- **Changes**:
  - Modified `find_checkpoint_files()` to search both root and subdirectories
  - Backward compatible with existing flat structure
  - Supports new organized directory structure

### 5. Demo Script Updates
- **File**: `demo_train.py`
- **Changes**: Updated to use model-specific directory structure

## New File Outputs

### During Training
Each model training session now generates:
1. **Model checkpoint**: `{model_type}_{timestamp}.pt`
2. **Training history**: `{model_type}_{timestamp}_history.json`
3. **Training curves**: `{model_type}_{timestamp}_curves.png`
4. **Forecast trajectory**: `{model_type}_{timestamp}_trajectory.png` *(NEW)*

### Trajectory Plot Features
- **Input Context**: Shows the input sequence used for initialization (black line)
- **True Target**: Shows the actual target trajectory (blue line with circles)
- **Model Forecast**: Shows the model's predicted trajectory (red dashed line with squares)
- **Reference Lines**: Includes ENSO thresholds (±0.5°C) and zero line
- **Multiple Samples**: Shows up to 3 validation samples for robustness

## Usage Examples

### Training a Model
```bash
# Standard training (creates physics_informed/ subdirectory)
python train.py --model_type physics_informed --n_epochs 50

# ENSO-only training (creates physics_informed_enso_only/ subdirectory)
python train.py --model_type physics_informed --enso_only --n_epochs 50
```

### Directory Structure After Training
```
checkpoints/
├── physics_informed/
│   ├── physics_informed_20250116_143022.pt
│   ├── physics_informed_20250116_143022_history.json
│   ├── physics_informed_20250116_143022_curves.png
│   └── physics_informed_20250116_143022_trajectory.png
└── physics_informed_enso_only/
    ├── physics_informed_enso_only_20250116_144515.pt
    ├── physics_informed_enso_only_20250116_144515_history.json
    ├── physics_informed_enso_only_20250116_144515_curves.png
    └── physics_informed_enso_only_20250116_144515_trajectory.png
```

## Backward Compatibility
- Existing comparison scripts work with both old (flat) and new (organized) directory structures
- Legacy checkpoints in the root `checkpoints/` directory are still found and used
- No breaking changes to existing functionality

## Benefits
1. **Organization**: Clear separation of different model types and variants
2. **Visualization**: Immediate visual feedback on model forecast quality
3. **Comparison**: Easy to compare trajectory plots across different models
4. **Debugging**: Trajectory plots help identify training issues early
5. **Documentation**: Each model run is self-contained with all outputs

## Technical Details

### Trajectory Plot Generation
- Triggered when a new best validation loss is achieved
- Uses validation data to ensure unbiased evaluation
- Focuses on ENSO variable as the primary climate index of interest
- Handles both stochastic and deterministic models appropriately
- Error handling prevents training interruption if plotting fails

### Directory Naming Convention
- Base model type (e.g., `physics_informed`)
- Suffix for variants (e.g., `_enso_only` for ENSO-only training)
- Timestamp format: `YYYYMMDD_HHMMSS`
- Consistent across all output files

This implementation provides a comprehensive solution for organized model training with immediate visual feedback on forecast quality, making it easier to track, compare, and analyze different model variants.
