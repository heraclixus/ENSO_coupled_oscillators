# Model Comparison Scripts

This directory contains two comparison scripts for evaluating Neural ODE models against XRO:

## 🎯 **Primary Script: `simple_enso_comparison.py`** (RECOMMENDED)

**Use this script for reliable ENSO forecast skill comparison.**

### Features:
- ✅ **Works reliably** with current environment
- ✅ **ENSO-focused analysis** (Nino3.4 forecast skills)
- ✅ **Fast execution** (~2-3 minutes)
- ✅ **Clear visualizations** with working plots
- ✅ **Simple correlation metrics** that are easy to interpret
- ✅ **Supports all model variants** including ENSO-only training

### Usage:
```bash
# Run the comparison (requires trained models)
python simple_enso_comparison.py
```

### Output:
- `enso_forecast_skills_simple.png` - ENSO correlation skill plot
- Console output with skill summary at 3, 6, and 12-month leads

### Model Variants Compared:
- **XRO** (baseline)
- **PhysicsNODE_external** (multivariate training)
- **PhysicsNODE_external_ENSO** (ENSO-only training)
- Plus any other trained variants found in checkpoints/

---

## 🔧 **Advanced Script: `trained_model_comparison_advanced.py`** (EXPERIMENTAL)

**Advanced comparison with full climpred integration - currently has API compatibility issues.**

### Status:
- ⚠️ **Currently broken** due to climpred API changes
- 🔄 **For future development** when climpred issues are resolved
- 📊 **More comprehensive metrics** when working

### Issues:
- climpred API has changed significantly in v1.1.0
- Coordinate structure compatibility problems
- Complex skill calculation pipeline

### Potential Future Features:
- Multiple skill metrics (correlation, RMSE, etc.)
- Monthly grouping analysis
- More sophisticated verification periods
- Advanced statistical measures

---

## 📋 **Training Models for Comparison**

Before running comparisons, train models with ENSO-only variants:

```bash
# Standard multivariate training
python train.py --model_type physics_informed --n_epochs 50

# ENSO-only training (NEW!)
python train.py --model_type physics_informed --enso_only --n_epochs 50

# Other variants
python train.py --model_type neural_ode --n_epochs 50
python train.py --model_type neural_ode --enso_only --n_epochs 50
python train.py --model_type stochastic --n_epochs 50
python train.py --model_type stochastic --enso_only --n_epochs 50
```

## 🎯 **Key Research Question**

The comparison addresses: **"Does training with ENSO-only loss improve ENSO forecasting compared to multivariate loss?"**

**Current Results**: ENSO-only training shows better short-term skill (3-month) but worse long-term skill (12-month), suggesting the importance of coupled dynamics for extended forecasting.

---

## 📁 **File Structure**

```
XRO/
├── simple_enso_comparison.py              # ← USE THIS
├── trained_model_comparison_advanced.py   # ← EXPERIMENTAL
├── train.py                               # Training script with --enso_only flag
├── checkpoints/                           # Trained model checkpoints
│   ├── physics_informed_*.pt             # Multivariate models
│   └── physics_informed_enso_only_*.pt   # ENSO-only models
└── enso_forecast_skills_simple.png       # Generated comparison plot
```
