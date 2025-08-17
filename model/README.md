# Neural ODE Climate Models

This directory contains neural ODE implementations inspired by the XRO (eXtended Recharge Oscillator) model for climate dynamics forecasting. We provide multiple model variants with different levels of physics constraints and noise handling approaches.

## Model Overview

| Model | Physics Level | Oscillator Guarantee | Noise Modes | Graph Structure | Key Features |
|-------|---------------|---------------------|-------------|-----------------|--------------|
| **NeuralODE** | Low | ❌ | External | None | General neural dynamics |
| **PhysicsInformedODE** | Medium | ❌ | External | None | XRO-inspired structure |
| **StochasticNeuralODE** | Low | ❌ | Internal | None | Noise in vector field |
| **OscillatorNeuralODE** | High | ✅ | Both | None | **Guaranteed oscillations** |
| **GraphNeuralODE** | Low | ❌ | Both | **Fully-connected** | **Graph-based dynamics** |
| **PhysicsGraphNeuralODE** | Medium-High | ❌ | Both | **Fully-connected** | **Physics + Graph learning** |
| **SineGraphNeuralODE** | Low | ❌ | Both | **Fully-connected** | **Kuramoto-inspired coupling** |
| **SinePhysicsGraphNeuralODE** | Medium-High | ❌ | Both | **Fully-connected** | **Physics + Kuramoto coupling** |

## Models

### 1. NeuralODE (`neural_ode.py`)
**General neural ODE with flexible architecture**

**Mathematical Form:**
```
dx/dt = NN_linear(x, s(t)) + NN_nonlinear(x, s(t)) + ε_external(x, t)
```

Where:
- `s(t)`: Seasonal embedding (Fourier features)
- `NN_linear`: Neural network producing state-dependent linear dynamics
- `NN_nonlinear`: Neural network for nonlinear terms with ENSO-specific components
- `ε_external`: Post-integration noise (external mode only)

**Key Features:**
- **Seasonal embedding**: Learnable seasonal cycle representation
- **Physics-informed blocks**: Neural networks that incorporate climate physics insights
- **ENSO-specific terms**: Special handling for ENSO T and H variables with nonlinear interactions (T², TH, T³, T²H, TH²)
- **State-dependent noise**: Learnable noise model that depends on current state and season

### 2. PhysicsInformedODE (`physics_informed_ode.py`)
**Structured model closely following XRO's mathematical framework**

**Mathematical Form:**
```
dx/dt = L(t) * x + N_quad(x) + N_cubic(x) + N_ENSO(x) + ε_external(x, t)
```

Where:
- `L(t) = L₀ + Σₖ [Lₖᶜcos(kωt) + Lₖˢsin(kωt)]`: Seasonal linear operator
- `N_quad(x)`: Quadratic nonlinear terms
- `N_cubic(x)`: Cubic nonlinear terms  
- `N_ENSO(x)`: ENSO-specific nonlinear interactions
- `ε_external`: Seasonal and state-dependent noise

**Key Features:**
- **Seasonal linear operator**: Explicit Fourier decomposition like XRO
- **Hierarchical nonlinearity**: Separate networks for different polynomial orders
- **ENSO physics**: Dedicated treatment of recharge oscillator dynamics
- **Interpretable structure**: Direct correspondence to XRO components

### 3. StochasticNeuralODE (`stochastic_neural_ode.py`)
**Neural ODE with noise directly in the vector field (SDE-like)**

**Mathematical Form:**
```
dx/dt = NN_dynamics(x, s(t)) + σ(t) * ε_internal(t)
```

Where:
- `NN_dynamics`: Deterministic neural ODE function
- `σ(t)`: Seasonal noise amplitude
- `ε_internal`: Random vector added directly to dx/dt (internal noise)

**Key Features:**
- **Internal noise**: Stochastic forcing within the differential equation
- **Seasonal noise modulation**: Time-dependent noise amplitude
- **SDE-like behavior**: Noise affects the dynamics continuously
- **Flexible base**: Can use any neural ODE as the deterministic component

### 4. OscillatorNeuralODE (`oscillator_neural_ode.py`) ⭐
**Neural ODE with explicit oscillator constraints for guaranteed ENSO-like behavior**

**Mathematical Form:**
```
dx/dt = L_NN(x, s(t)) * x + N_NN(x, s(t)) + O_explicit(x, t) + ε(x, t)
```

**Explicit Oscillator Component:**
```
O_explicit: d²q/dt² + 2γ*dq/dt + ω²*q = F_NN(x)
         → dq/dt = v
           dv/dt = -2γ*v - ω²*q + F_NN(x)
         → O_explicit(x,t) = C_pos * q + C_vel * v
```

Where:
- `q, v`: Oscillator position and velocity states
- `ω`: Learnable oscillation frequency (initialized for ENSO periods: 3-4 years)
- `γ`: Learnable damping coefficient
- `F_NN(x)`: Neural network forcing from climate state
- `C_pos, C_vel`: Coupling matrices connecting oscillators to climate variables
- `s(t)`: Seasonal modulation of oscillator strength

**Noise Modes:**
- **External**: `ε_external(x, t)` added after ODE integration
- **Internal**: `ε_internal(t)` added directly to dx/dt

**Key Features:**
- **Guaranteed Oscillations**: Explicit harmonic oscillators ensure complex eigenvalues
- **ENSO-Appropriate Periods**: Initialized and constrained for 2-7 year timescales
- **Learnable Parameters**: Frequencies (ω) and damping (γ) adapt during training
- **Physics-Informed Losses**: Spectral, amplitude, and smoothness regularization
- **Dual Noise Support**: Both external and internal noise modes available

### 5. GraphNeuralODE (`graph_neural_ode.py`) 🕸️
**Graph Neural ODE using GCN or GAT layers for explicit variable interactions**

**Mathematical Form:**
```
dx/dt = GNN(x, A, s(t)) + ε(x, t)
```

Where:
- `GNN`: Graph Neural Network (GCN or GAT layers)
- `A`: Fully-connected adjacency matrix (all variables interact)
- `s(t)`: Seasonal embedding (Fourier features)
- `ε(x, t)`: Noise (external or internal modes)

**Key Features:**
- **Graph Structure**: Fully-connected graph representing climate variable interactions
- **GNN Layers**: Choice of GCN (spectral) or GAT (attention-based) convolutions
- **Explicit Interactions**: Graph structure explicitly models how variables influence each other
- **ENSO-Specific Terms**: Special nonlinear terms for first two variables (T, H)
- **Dual Noise Support**: Both external and internal noise modes
- **Scalable Architecture**: Graph operations scale well with number of variables

### 6. PhysicsGraphNeuralODE (`physics_graph_ode.py`) 🔬🕸️
**Physics-informed Graph Neural ODE combining XRO structure with Graph Neural Networks**

**Mathematical Form:**
```
dx/dt = L(t) * x + N_graph(x) + ε(x, t)
```

**Components:**
- `L(t) = L₀ + Σₖ [Lₖᶜcos(kωt) + Lₖˢsin(kωt)]`: Seasonal linear operator (like XRO)
- `N_graph(x)`: Graph-based nonlinear terms using GNN layers
- `ε(x, t)`: Graph-based noise model with seasonal dependence

**Graph-Based Nonlinear Terms:**
```
N_graph(x) = GNN_quad(x) + GNN_cubic(x) + GNN_ENSO(x)
```

Where:
- `GNN_quad(x)`: Graph network for quadratic terms
- `GNN_cubic(x)`: Graph network for cubic terms  
- `GNN_ENSO(x)`: Graph networks for ENSO-specific interactions (T², TH, T³, T²H, TH²)

**Key Features:**
- **Physics + Graph Learning**: Combines XRO's mathematical structure with graph learning
- **Structured Nonlinearity**: Separate graph networks for different polynomial orders
- **ENSO Physics**: Dedicated graph networks for recharge oscillator dynamics
- **Interpretable Structure**: Maintains physical meaning while adding graph flexibility
- **Graph-Based Noise**: Seasonal and state-dependent noise using graph networks
- **Best of Both Worlds**: Physical constraints + flexible graph representations

### 7. SineGraphNeuralODE (`sine_graph_neural_ode.py`) 🌊🕸️
**Graph Neural ODE with sine-based message passing inspired by Kuramoto oscillators**

**Mathematical Form:**
```
dx/dt = SineGNN(x, A, s(t)) + ε(x, t)
```

**Sine Message Passing:**
```
m_ij = sin(NN_msg(x_j) - NN_phase(x_i)) * NN_weight(x_i, x_j)
```

**Kuramoto-Inspired Dynamics:**
```
dθ_i/dt = ω_i + Σ_j K_ij * sin(θ_j - θ_i)
```

Where:
- `SineGNN`: Sine aggregation Graph Neural Network
- `NN_msg, NN_phase`: Neural networks producing phase-like representations
- `NN_weight`: Neural network producing coupling weights
- `ω_i`: Learnable natural frequencies
- `K_ij`: Learnable coupling strengths

**Key Features:**
- **Kuramoto Oscillator Inspiration**: Sine-based coupling between climate variables
- **Phase-Like Representations**: Neural networks learn phase relationships
- **Bounded Interactions**: Sine functions provide stable, bounded coupling
- **Two Aggregation Types**: 'sine' (multi-layer) and 'kuramoto' (direct oscillator)
- **Dual Noise Support**: Both external and internal noise modes
- **Synchronization Dynamics**: Natural tendency toward phase-locked climate modes

### 8. SinePhysicsGraphNeuralODE (`sine_physics_graph_ode.py`) 🔬🌊🕸️
**Physics-informed Graph Neural ODE with Kuramoto-inspired sine aggregation**

**Mathematical Form:**
```
dx/dt = L(t) * x + N_sine_graph(x) + ε(x, t)
```

**Components:**
- `L(t) = L₀ + Σₖ [Lₖᶜcos(kωt) + Lₖˢsin(kωt)]`: Seasonal linear operator (like XRO)
- `N_sine_graph(x)`: Sine-based graph nonlinear terms
- `ε(x, t)`: Sine-based graph noise model

**Sine-Based Nonlinear Terms:**
```
N_sine_graph(x) = SineGNN_quad(x) + SineGNN_cubic(x) + SineGNN_ENSO(x)
```

Where:
- `SineGNN_quad(x)`: Sine aggregation for quadratic terms
- `SineGNN_cubic(x)`: Sine aggregation for cubic terms
- `SineGNN_ENSO(x)`: Sine aggregation for ENSO-specific interactions

**Key Features:**
- **Physics + Kuramoto Coupling**: Combines XRO structure with sine-based interactions
- **Oscillator-Inspired Nonlinearity**: All nonlinear terms use sine aggregation
- **ENSO Phase Dynamics**: Dedicated sine networks for T and H phase relationships
- **Seasonal Modulation**: Time-dependent sine coupling strengths
- **Interpretable + Flexible**: Physical meaning with oscillator-inspired interactions
- **Best of Four Worlds**: Physics + Graph + Neural + Oscillator dynamics

## Mathematical Framework

All models follow the general ODE structure inspired by XRO but with different levels of physics constraints:

### General Form
```
dx/dt = L(t) * x + N(x) + ξ(t)
```

Where:
- `L(t)`: Seasonally-varying linear operator (neural or explicit)
- `N(x)`: Nonlinear terms (learned by neural networks)
- `ξ(t)`: Stochastic noise term (external or internal)

### Model-Specific Implementations

#### NeuralODE: Fully Neural Approach
```
dx/dt = NN([x; s(t)]) + ε_external
```
- **Pros**: Maximum flexibility, can learn arbitrary dynamics
- **Cons**: No guarantee of physical behavior or oscillations

#### PhysicsInformedODE: XRO-Structured Approach
```
dx/dt = [L₀ + Σₖ Lₖᶜcos(kωt) + Lₖˢsin(kωt)] * x + NN_quad(x) + NN_cubic(x) + ε_external
```
- **Pros**: Interpretable structure, explicit seasonal cycles
- **Cons**: May lose oscillations during training

#### StochasticNeuralODE: SDE-like Approach
```
dx/dt = NN_deterministic([x; s(t)]) + σ(t) * ε_internal
```
- **Pros**: Continuous stochastic forcing, realistic noise
- **Cons**: No oscillation guarantees, complex training

#### OscillatorNeuralODE: Guaranteed Oscillations ⭐
```
dx/dt = NN_linear([x; s(t)]) * x + NN_nonlinear([x; s(t)]) + O_harmonic(t) + ε
```

**Harmonic Oscillator Component:**
```
For each oscillator i:
  dqᵢ/dt = vᵢ
  dvᵢ/dt = -2γᵢ*vᵢ - ωᵢ²*qᵢ + Fᵢ_NN(x)
  
Contribution to climate state:
  O_harmonic = Σᵢ [Cᵢ_pos * qᵢ + Cᵢ_vel * vᵢ] * m(t)
```

Where:
- `qᵢ, vᵢ`: Position and velocity of oscillator i
- `ωᵢ`: Learnable frequency (initialized for ENSO: ωᵢ = 2π/Tᵢ, Tᵢ ∈ [36, 48] months)
- `γᵢ`: Learnable damping coefficient
- `Cᵢ_pos, Cᵢ_vel`: Learnable coupling matrices
- `m(t)`: Seasonal modulation factor
- `Fᵢ_NN(x)`: Neural network forcing from climate state

#### GraphNeuralODE: Graph-Based Dynamics 🕸️
```
dx/dt = GNN([x; s(t)], A) + ε
```
- **Pros**: Explicit variable interactions, scalable graph operations, attention mechanisms (GAT)
- **Cons**: No physics constraints, no oscillation guarantees

#### PhysicsGraphNeuralODE: Physics + Graph Learning 🔬🕸️
```
dx/dt = [L₀ + Σₖ Lₖᶜcos(kωt) + Lₖˢsin(kωt)] * x + GNN_nonlinear(x, A) + ε
```
- **Pros**: Combines XRO structure with graph learning, interpretable + flexible
- **Cons**: More complex architecture, may lose oscillations during training

#### SineGraphNeuralODE: Kuramoto-Inspired Dynamics 🌊🕸️
```
dx/dt = SineGNN([x; s(t)], A) + ε
```
**Sine Message Passing**: `m_ij = sin(NN_msg(x_j) - NN_phase(x_i)) * NN_weight(x_i, x_j)`
- **Pros**: Oscillator-inspired coupling, bounded interactions, phase relationships
- **Cons**: No physics constraints, complex sine aggregation

#### SinePhysicsGraphNeuralODE: Physics + Kuramoto Coupling 🔬🌊🕸️
```
dx/dt = [L₀ + Σₖ Lₖᶜcos(kωt) + Lₖˢsin(kωt)] * x + SineGNN_nonlinear(x, A) + ε
```
- **Pros**: Physics structure + oscillator coupling, interpretable phase dynamics
- **Cons**: Most complex architecture, requires careful tuning

## Enforcing Oscillatory Behavior

### The Oscillation Problem
Standard neural ODEs may lose oscillatory behavior during training because:
1. **Eigenvalue drift**: Linear operators can evolve to have only real eigenvalues
2. **Damping dominance**: Learned dynamics may become over-damped
3. **Loss function bias**: MSE loss doesn't explicitly encourage oscillations
4. **Gradient issues**: Oscillatory solutions can have unstable gradients

### OscillatorNeuralODE Solution ⭐

#### 1. **Explicit Harmonic Oscillators**
```python
# Guaranteed complex eigenvalues through harmonic oscillator physics
class HarmonicOscillatorBlock(nn.Module):
    def __init__(self, n_oscillators=2):
        # Learnable parameters (log-space for positivity)
        self.log_frequencies = nn.Parameter(torch.log(torch.tensor([2π/48, 2π/36])))  # 4-year, 3-year
        self.log_damping = nn.Parameter(torch.log(torch.tensor([0.1, 0.05])))
        
    def forward(self, t, x):
        # Solve: d²q/dt² + 2γ*dq/dt + ω²*q = F(x)
        # This GUARANTEES complex eigenvalues: λ = -γ ± i*sqrt(ω² - γ²)
```

#### 2. **Physics-Informed Loss Functions**
```python
def create_oscillator_loss(model, target_periods=[36, 48]):
    def loss_fn(trajectory, time):
        losses = []
        
        # Spectral loss: encourage target frequencies
        current_periods = 2π / exp(model.log_frequencies)
        period_loss = mean((current_periods - target_periods)²)
        losses.append(period_loss)
        
        # Amplitude regularization: prevent oscillator death
        osc_amplitude = norm(model.oscillator_state)
        amplitude_loss = exp(-osc_amplitude)
        losses.append(amplitude_loss)
        
        # Smoothness loss: encourage smooth oscillations
        acceleration = diff(diff(trajectory))
        smoothness_loss = mean(acceleration²)
        losses.append(smoothness_loss)
        
        # Autocorrelation loss: encourage periodic patterns
        for lag in [3, 6, 9]:  # months
            correlation = corrcoef(trajectory[:-lag], trajectory[lag:])
            autocorr_loss = clamp(-correlation, 0, 1)  # penalize negative correlations
            losses.append(autocorr_loss)
            
        return sum(losses)
```

#### 3. **Initialization Strategy**
```python
# Initialize for ENSO-appropriate periods
init_frequencies = [2π/48, 2π/36]  # 4-year and 3-year periods
init_damping = [0.1, 0.05]         # Light damping for sustained oscillations

# Coupling matrices initialized to connect oscillators to ENSO variables
position_coupling = nn.Linear(n_oscillators, state_dim)  # q → climate state
velocity_coupling = nn.Linear(n_oscillators, state_dim)  # v → climate state
```

#### 4. **Seasonal Modulation**
```python
# Oscillator strength varies seasonally (like ENSO's seasonal cycle)
seasonal_embedding = SeasonalEmbedding(t)
modulation_factor = sigmoid(NN_modulation(seasonal_embedding))
oscillator_contribution *= modulation_factor
```

#### 5. **Eigenvalue Analysis**
```python
def compute_eigenvalues(model, state):
    # Compute Jacobian of dx/dt with respect to x
    jacobian = autograd.functional.jacobian(model.ode_func, state)
    eigenvals = scipy.linalg.eigvals(jacobian)
    
    # Check for complex eigenvalues (oscillatory modes)
    complex_eigs = eigenvals[np.iscomplex(eigenvals)]
    oscillation_periods = 2π / np.abs(np.imag(complex_eigs))
    
    return oscillation_periods
```

### Comparison: Oscillation Guarantees

| Model | Oscillation Method | Guarantee | ENSO Periods |
|-------|-------------------|-----------|--------------|
| **NeuralODE** | Emergent from training | ❌ None | May disappear |
| **PhysicsInformedODE** | XRO-like structure | ⚠️ Weak | May drift |
| **StochasticNeuralODE** | Noise-driven | ❌ None | Stochastic only |
| **OscillatorNeuralODE** | **Explicit oscillators** | ✅ **Strong** | **Guaranteed 2-7 years** |

### Why This Works

1. **Mathematical Guarantee**: Harmonic oscillators have complex eigenvalues by construction
2. **Learnable Parameters**: Frequencies and damping adapt to data while maintaining oscillatory structure
3. **Physics-Informed Losses**: Multiple loss terms encourage and maintain oscillatory behavior
4. **Robust Design**: Even if neural components fail, explicit oscillators ensure baseline oscillatory dynamics
5. **ENSO-Tuned**: Initialized and constrained for realistic ENSO timescales

## Key Features

### XRO-Inspired Design Principles
1. **Seasonal awareness**: Explicit modeling of seasonal cycles through Fourier decomposition
2. **Physical structure**: Separation of linear dynamics, nonlinear feedbacks, and stochastic forcing
3. **ENSO focus**: Special treatment of ENSO variables with guaranteed oscillatory behavior
4. **Hierarchical nonlinearity**: Multiple types of nonlinear terms for different physical processes

### Neural Network Enhancements
1. **Learnable parameters**: All components (linear operators, nonlinear terms, noise, oscillators) learned from data
2. **Flexible architecture**: Can handle arbitrary numbers of climate variables
3. **End-to-end training**: Optimized directly for forecasting performance using MSE + oscillator losses
4. **Oscillation guarantees**: Explicit harmonic oscillators ensure sustained ENSO-like behavior

## Usage

### Model Creation Examples

#### 1. General Neural ODE
```python
from model import NeuralODE

model = NeuralODE(
    state_dim=10,
    var_names=['Nino34', 'WWV', 'NPMM', 'SPMM', 'IOB', 'IOD', 'SIOD', 'TNA', 'ATL3', 'SASD'],
    hidden_dim=64
)
```

#### 2. Physics-Informed ODE
```python
from model import PhysicsInformedODE

model = PhysicsInformedODE(
    state_dim=10,
    var_names=['Nino34', 'WWV', 'NPMM', ...],
    hidden_dim=64,
    fourier_modes=2  # Annual and semi-annual cycles
)
```

#### 3. Stochastic Neural ODE
```python
from model import StochasticNeuralODE

model = StochasticNeuralODE(
    state_dim=10,
    var_names=['Nino34', 'WWV', 'NPMM', ...],
    noise_scale=0.1,
    hidden_dim=64
)
```

#### 4. Oscillator Neural ODE ⭐
```python
from model import OscillatorNeuralODE

# External noise mode
model_ext = OscillatorNeuralODE(
    state_dim=10,
    var_names=['Nino34', 'WWV', 'NPMM', ...],
    noise_mode='external',
    hidden_dim=64
)

# Internal noise mode  
model_int = OscillatorNeuralODE(
    state_dim=10,
    var_names=['Nino34', 'WWV', 'NPMM', ...],
    noise_mode='internal',
    noise_scale=0.1,
    hidden_dim=64
)
```

#### 5. Graph Neural ODE 🕸️
```python
from model import GraphNeuralODE

# GCN-based model with external noise
model_gcn = GraphNeuralODE(
    state_dim=10,
    var_names=['Nino34', 'WWV', 'NPMM', ...],
    hidden_dim=64,
    gnn_type='gcn',
    noise_mode='external'
)

# GAT-based model with internal noise
model_gat = GraphNeuralODE(
    state_dim=10,
    var_names=['Nino34', 'WWV', 'NPMM', ...],
    hidden_dim=64,
    gnn_type='gat',
    noise_mode='internal',
    noise_scale=0.1
)
```

#### 6. Physics Graph Neural ODE 🔬🕸️
```python
from model import PhysicsGraphNeuralODE

# Physics-informed GCN model
model_phys_gcn = PhysicsGraphNeuralODE(
    state_dim=10,
    var_names=['Nino34', 'WWV', 'NPMM', ...],
    hidden_dim=64,
    gnn_type='gcn',
    fourier_modes=2,  # Annual and semi-annual cycles
    noise_mode='external'
)

# Physics-informed GAT model with internal noise
model_phys_gat = PhysicsGraphNeuralODE(
    state_dim=10,
    var_names=['Nino34', 'WWV', 'NPMM', ...],
    hidden_dim=64,
    gnn_type='gat',
    fourier_modes=2,
    noise_mode='internal',
    noise_scale=0.1
)
```

#### 7. Sine Graph Neural ODE 🌊🕸️
```python
from model import SineGraphNeuralODE

# Sine aggregation model with external noise
model_sine = SineGraphNeuralODE(
    state_dim=10,
    var_names=['Nino34', 'WWV', 'NPMM', ...],
    hidden_dim=64,
    sine_type='sine',  # Multi-layer sine aggregation
    noise_mode='external'
)

# Kuramoto aggregation model with internal noise
model_kuramoto = SineGraphNeuralODE(
    state_dim=10,
    var_names=['Nino34', 'WWV', 'NPMM', ...],
    hidden_dim=64,
    sine_type='kuramoto',  # Direct Kuramoto dynamics
    noise_mode='internal',
    noise_scale=0.1
)
```

#### 8. Sine Physics Graph Neural ODE 🔬🌊🕸️
```python
from model import SinePhysicsGraphNeuralODE

# Physics + Sine aggregation model
model_phys_sine = SinePhysicsGraphNeuralODE(
    state_dim=10,
    var_names=['Nino34', 'WWV', 'NPMM', ...],
    hidden_dim=64,
    sine_type='sine',
    fourier_modes=2,  # Annual and semi-annual cycles
    noise_mode='external'
)

# Physics + Kuramoto aggregation model with internal noise
model_phys_kuramoto = SinePhysicsGraphNeuralODE(
    state_dim=10,
    var_names=['Nino34', 'WWV', 'NPMM', ...],
    hidden_dim=64,
    sine_type='kuramoto',
    fourier_modes=2,
    noise_mode='internal',
    noise_scale=0.1
)
```

### Simulation (like XRO.simulate)

#### External Noise Models (NeuralODE, PhysicsInformedODE, OscillatorNeuralODE-external)
```python
simulation = model.simulate(
    x0_data=initial_conditions,  # xarray Dataset
    nyear=10,                    # Simulate 10 years
    ncopy=100,                   # 100 ensemble members
    add_noise=True,              # Include stochastic forcing (external)
    device='cpu'
)
```

#### Internal Noise Models (StochasticNeuralODE, OscillatorNeuralODE-internal)
```python
simulation = model.simulate(
    x0_data=initial_conditions,  # xarray Dataset
    nyear=10,                    # Simulate 10 years
    ncopy=100,                   # 100 ensemble members
    enable_noise=True,           # Enable internal noise in vector field
    device='cpu'
)
```

### Forecasting (like XRO.reforecast)

#### External Noise Models
```python
forecasts = model.reforecast(
    init_data=initialization_data,  # xarray Dataset with time dimension
    n_month=21,                     # 21-month forecasts
    ncopy=50,                       # 50 ensemble members
    add_noise=False,                # Deterministic for skill evaluation
    device='cpu'
)
```

#### Internal Noise Models
```python
forecasts = model.reforecast(
    init_data=initialization_data,  # xarray Dataset with time dimension
    n_month=21,                     # 21-month forecasts
    ncopy=50,                       # 50 ensemble members
    enable_noise=False,             # Deterministic for skill evaluation
    device='cpu'
)
```

### Oscillator Analysis (OscillatorNeuralODE only)
```python
# Get oscillator information
osc_info = model.get_oscillator_info()
print(f"Oscillation periods: {osc_info['periods_years']} years")
print(f"Frequencies: {osc_info['frequencies']} rad/month")
print(f"Damping: {osc_info['damping']}")

# Compute eigenvalues at a given state
eigenvals = model.compute_eigenvalues_at_state(torch.zeros(state_dim))
complex_eigs = eigenvals[np.iscomplex(eigenvals)]
print(f"Complex eigenvalues: {complex_eigs}")
```

## Training

Use the `train.py` script to train all model variants:

### Basic Model Training
```bash
# Train Neural ODE
python train.py --model_type neural_ode --n_epochs 100 --batch_size 32

# Train Physics-Informed ODE
python train.py --model_type physics_informed --n_epochs 100 --batch_size 32

# Train Stochastic Neural ODE
python train.py --model_type stochastic --n_epochs 100 --batch_size 32

# Train Oscillator Neural ODE (external noise)
python train.py --model_type oscillator_external --n_epochs 100 --batch_size 32

# Train Oscillator Neural ODE (internal noise)
python train.py --model_type oscillator_internal --n_epochs 100 --batch_size 32

# Train Graph Neural ODE (GCN)
python train.py --model_type graph_gcn --n_epochs 100 --batch_size 32

# Train Graph Neural ODE (GAT)
python train.py --model_type graph_gat --n_epochs 100 --batch_size 32

# Train Physics Graph Neural ODE (GCN)
python train.py --model_type physics_graph_gcn --n_epochs 100 --batch_size 32

# Train Physics Graph Neural ODE (GAT)
python train.py --model_type physics_graph_gat --n_epochs 100 --batch_size 32

# Train Sine Graph Neural ODE (Sine aggregation)
python train.py --model_type sine_graph_sine --n_epochs 100 --batch_size 32

# Train Sine Graph Neural ODE (Kuramoto aggregation)
python train.py --model_type sine_graph_kuramoto --n_epochs 100 --batch_size 32

# Train Sine Physics Graph Neural ODE (Sine aggregation)
python train.py --model_type sine_physics_graph_sine --n_epochs 100 --batch_size 32

# Train Sine Physics Graph Neural ODE (Kuramoto aggregation)
python train.py --model_type sine_physics_graph_kuramoto --n_epochs 100 --batch_size 32
```

### ENSO-Only Training
```bash
# Focus on ENSO variables only (faster training, ENSO-specific)
python train.py --model_type neural_ode --enso_only --n_epochs 100
python train.py --model_type physics_informed --enso_only --n_epochs 100
python train.py --model_type oscillator_external --enso_only --n_epochs 100
python train.py --model_type graph_gcn --enso_only --n_epochs 100
python train.py --model_type graph_gat --enso_only --n_epochs 100
python train.py --model_type physics_graph_gcn --enso_only --n_epochs 100
python train.py --model_type physics_graph_gat --enso_only --n_epochs 100
python train.py --model_type sine_graph_sine --enso_only --n_epochs 100
python train.py --model_type sine_graph_kuramoto --enso_only --n_epochs 100
python train.py --model_type sine_physics_graph_sine --enso_only --n_epochs 100
python train.py --model_type sine_physics_graph_kuramoto --enso_only --n_epochs 100
```

### Training with Oscillator Loss (OscillatorNeuralODE)
```python
from model.oscillator_neural_ode import create_oscillator_loss

# Create model
model = OscillatorNeuralODE(state_dim=10, var_names=var_names)

# Create oscillator loss
osc_loss_fn = create_oscillator_loss(model, target_periods=[36, 48])  # 3-4 year periods

# Training loop
for batch in dataloader:
    # Standard forecasting loss
    forecast_loss = F.mse_loss(model_output, target)
    
    # Oscillator loss
    oscillator_loss = osc_loss_fn(trajectory, time_points)
    
    # Combined loss
    total_loss = forecast_loss + 0.1 * oscillator_loss  # Weight oscillator loss
    total_loss.backward()
```

## Evaluation

### Individual Model Evaluation
```bash
# Evaluate specific model checkpoints
python evaluate_neural_ode.py \
    --checkpoint checkpoints/physics_informed_20240101_120000.pt \
    --model_type physics_informed \
    --n_month 21

python evaluate_neural_ode.py \
    --checkpoint checkpoints/oscillator_external_20240101_120000.pt \
    --model_type oscillator_external \
    --n_month 21

python evaluate_neural_ode.py \
    --checkpoint checkpoints/graph_gcn_20240101_120000.pt \
    --model_type graph_gcn \
    --n_month 21

python evaluate_neural_ode.py \
    --checkpoint checkpoints/physics_graph_gat_20240101_120000.pt \
    --model_type physics_graph_gat \
    --n_month 21
```

### Comprehensive Model Comparison
```bash
# Compare all trained models automatically
python trained_model_comparison.py

# Generates:
# - ENSO forecast skill plots for all model variants
# - Performance comparison table
# - Skill metrics at different lead times
```

### Oscillatory Behavior Analysis
```bash
# Analyze oscillatory properties of all models
python simple_oscillatory_analysis.py

# Generates:
# - Eigenvalue analysis in complex plane
# - Oscillation period comparison
# - Phase portraits and spectral analysis
```

## Testing

Run test scripts to verify functionality:

```bash
# Test basic Neural ODE models
python test_neural_ode.py

# Test Stochastic Neural ODE
python test_stochastic_ode.py

# Test Oscillator Neural ODE variants
python test_oscillator_neural_ode.py

# Test oscillator variants (external vs internal)
python test_oscillator_variants.py

# Test Graph Neural ODE models
python test_graph_neural_ode.py

# Test Sine Graph Neural ODE models
python test_sine_graph_neural_ode.py
```

## Model Architecture Details

### PhysicsInformedODE Components

1. **SeasonalLinearOperator**: 
   - Implements L(t) with Fourier coefficients
   - Learnable parameters for annual mean, annual cycle, semi-annual cycle, etc.

2. **NonlinearTerms**:
   - General quadratic and cubic networks
   - ENSO-specific networks for T and H equations
   - Handles the nonlinear interactions: T², TH, T³, T²H, TH²

3. **NoiseModel**:
   - Seasonal noise amplitude (like XRO's ξ_stdac)
   - State-dependent noise scaling

### Training Strategy

1. **Sequence-to-sequence learning**: Model learns to predict future states from past sequences
2. **Lead time weighting**: Loss function emphasizes shorter forecast leads
3. **MSE optimization**: Direct optimization for forecast accuracy
4. **Gradient clipping**: Prevents exploding gradients during ODE integration

## Advantages over Traditional XRO

### All Neural ODE Models
1. **Data-driven**: Parameters learned directly from data rather than fitted analytically
2. **Flexible nonlinearity**: Neural networks can capture complex nonlinear relationships
3. **End-to-end optimization**: Optimized directly for forecasting performance
4. **Scalable**: Can handle more variables and longer sequences

### Model-Specific Advantages

#### NeuralODE
- **Maximum flexibility**: Can learn arbitrary dynamics without constraints
- **Fast training**: Simple architecture with fewer parameters
- **General applicability**: Works for any dynamical system

#### PhysicsInformedODE
- **Interpretable structure**: Direct correspondence to XRO components
- **Seasonal awareness**: Explicit Fourier decomposition like XRO
- **Physical constraints**: Maintains XRO's mathematical framework

#### StochasticNeuralODE
- **Realistic noise**: Continuous stochastic forcing within dynamics
- **SDE-like behavior**: More realistic than post-integration noise
- **Flexible stochasticity**: Learnable noise patterns

#### OscillatorNeuralODE ⭐
- **Guaranteed oscillations**: Mathematical guarantee of ENSO-like behavior
- **Robust training**: Oscillations cannot be lost during optimization
- **ENSO-tuned**: Initialized and constrained for realistic timescales
- **Best of both worlds**: Combines neural flexibility with physical guarantees
- **Dual noise support**: Both external and internal noise modes
- **Physics-informed losses**: Multiple objectives ensure oscillatory behavior

#### GraphNeuralODE 🕸️
- **Explicit interactions**: Graph structure models variable relationships directly
- **Scalable architecture**: Graph operations scale well with number of variables
- **Attention mechanisms**: GAT variant learns which interactions are important
- **Inductive biases**: Graph structure provides better generalization
- **Dual noise support**: Both external and internal noise modes
- **Flexible GNN layers**: Choice between GCN (spectral) and GAT (attention)

#### PhysicsGraphNeuralODE 🔬🕸️
- **Physics + Graph learning**: Combines XRO structure with flexible graph networks
- **Interpretable structure**: Maintains physical meaning while adding graph flexibility
- **Structured nonlinearity**: Separate graph networks for different polynomial orders
- **ENSO physics**: Dedicated graph networks for recharge oscillator dynamics
- **Graph-based noise**: Seasonal and state-dependent noise using graph networks
- **Best of three worlds**: Physical constraints + graph learning + neural flexibility

#### SineGraphNeuralODE 🌊🕸️
- **Kuramoto oscillator inspiration**: Sine-based coupling inspired by collective synchronization
- **Phase relationships**: Can learn phase-locked and phase-shifted climate mode interactions
- **Bounded interactions**: Sine functions provide stable, bounded nonlinear coupling
- **Synchronization dynamics**: Natural tendency toward synchronized/anti-synchronized states
- **Two aggregation types**: Choice between multi-layer sine or direct Kuramoto dynamics
- **Dual noise support**: Both external and internal noise modes

#### SinePhysicsGraphNeuralODE 🔬🌊🕸️
- **Physics + Kuramoto coupling**: Combines XRO structure with oscillator-inspired interactions
- **Oscillator-informed nonlinearity**: All nonlinear terms use sine-based message passing
- **ENSO phase dynamics**: Dedicated sine networks for T and H phase relationships
- **Interpretable phase coupling**: Physical meaning with learnable phase interactions
- **Seasonal sine modulation**: Time-dependent oscillator coupling strengths
- **Best of four worlds**: Physical constraints + graph learning + neural flexibility + oscillator dynamics

## Dependencies

- PyTorch
- torchdiffeq (for ODE integration)
- torch-geometric (for Graph Neural Networks)
- xarray
- numpy
- matplotlib
- scipy (for eigenvalue analysis)

## References

The models are inspired by:
- Zhao, S., et al. (2024). Explainable El Niño predictability from climate mode interactions. Nature.
- Chen, R. T. Q., et al. (2018). Neural Ordinary Differential Equations. NeurIPS.
