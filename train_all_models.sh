#!/bin/bash

# Comprehensive Training Script for All Neural ODE Climate Models
# This script trains all model variants with both multivariate and ENSO-only configurations
# 
# Usage: ./train_all_models.sh [--epochs N] [--batch_size N] [--device cuda/cpu] [--enso_only] [--multivariate_only]
#
# Examples:
#   ./train_all_models.sh                           # Train all models with default settings
#   ./train_all_models.sh --epochs 200              # Train all models for 200 epochs
#   ./train_all_models.sh --enso_only               # Train only ENSO-only variants
#   ./train_all_models.sh --device cpu              # Force CPU training
#   ./train_all_models.sh --epochs 50 --enso_only  # Train ENSO-only models for 50 epochs

set -e  # Exit on any error

# Default parameters
EPOCHS=100
BATCH_SIZE=256
DEVICE="auto"  # Will auto-detect CUDA
TRAIN_MULTIVARIATE=true
TRAIN_ENSO_ONLY=true
CONDA_ENV="spde"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --enso_only)
            TRAIN_MULTIVARIATE=false
            shift
            ;;
        --multivariate_only)
            TRAIN_ENSO_ONLY=false
            shift
            ;;
        --conda_env)
            CONDA_ENV="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--epochs N] [--batch_size N] [--device cuda/cpu] [--enso_only] [--multivariate_only] [--conda_env ENV]"
            echo ""
            echo "Options:"
            echo "  --epochs N              Number of training epochs (default: 100)"
            echo "  --batch_size N          Batch size (default: 32)"
            echo "  --device cuda/cpu/auto  Training device (default: auto)"
            echo "  --enso_only             Train only ENSO-only variants"
            echo "  --multivariate_only     Train only multivariate variants"
            echo "  --conda_env ENV         Conda environment name (default: graph)"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${BLUE}================================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================================${NC}"
}

print_section() {
    echo -e "\n${PURPLE}>>> $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

# Function to activate conda environment
activate_conda() {
    print_section "Activating conda environment: $CONDA_ENV"
    
    # Initialize conda for bash
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        print_error "Could not find conda installation"
        exit 1
    fi
    
    # Activate environment
    conda activate "$CONDA_ENV" || {
        print_error "Failed to activate conda environment: $CONDA_ENV"
        print_info "Available environments:"
        conda env list
        exit 1
    }
    
    print_success "Activated conda environment: $CONDA_ENV"
    print_info "Python version: $(python --version)"
    print_info "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
}

# Function to check CUDA availability
check_cuda() {
    print_section "Checking CUDA availability"
    
    if [[ "$DEVICE" == "auto" ]]; then
        if python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            DEVICE="cuda"
            print_success "CUDA is available - using GPU acceleration"
            python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" 2>/dev/null || true
        else
            DEVICE="cpu"
            print_warning "CUDA not available - using CPU"
        fi
    else
        print_info "Device manually set to: $DEVICE"
    fi
}

# Function to train a single model
train_model() {
    local model_type=$1
    local enso_flag=$2
    local model_name=$3
    
    print_section "Training $model_name"
    
    local cmd="python train.py --model_type $model_type --n_epochs $EPOCHS --batch_size $BATCH_SIZE --device $DEVICE"
    if [[ "$enso_flag" == "true" ]]; then
        cmd="$cmd --enso_only"
    fi
    
    print_info "Command: $cmd"
    
    # Run training with error handling
    if eval "$cmd"; then
        print_success "Successfully trained $model_name"
    else
        print_error "Failed to train $model_name"
        return 1
    fi
}

# Function to estimate training time
estimate_time() {
    local total_models=0
    
    if [[ "$TRAIN_MULTIVARIATE" == "true" ]]; then
        total_models=$((total_models + 14))  # 14 multivariate model types
    fi
    
    if [[ "$TRAIN_ENSO_ONLY" == "true" ]]; then
        total_models=$((total_models + 14))  # 14 ENSO-only model types
    fi
    
    local estimated_minutes=$((total_models * EPOCHS / 10))  # Rough estimate: 10 epochs per minute
    print_info "Estimated training time: ~$estimated_minutes minutes for $total_models models"
    print_info "Training configuration: $EPOCHS epochs, batch size $BATCH_SIZE, device $DEVICE"
}

# Main training function
main() {
    print_header "COMPREHENSIVE NEURAL ODE CLIMATE MODEL TRAINING"
    print_info "Training all model variants with multiple configurations"
    print_info "Models: Neural ODE, Physics NODE, Stochastic NODE, Oscillator NODE, Graph NODE, Physics Graph NODE, Sine Graph NODE, Sine Physics Graph NODE"
    
    # Setup
    activate_conda
    check_cuda
    estimate_time
    
    # Create checkpoints directory
    mkdir -p checkpoints
    print_success "Created checkpoints directory"
    
    # Define all model types
    declare -a MODEL_TYPES=(
        "neural_ode"
        "physics_informed" 
        "stochastic"
        "oscillator_external"
        "oscillator_internal"
        "graph_gcn"
        "graph_gat"
        "physics_graph_gcn"
        "physics_graph_gat"
        "sine_graph_sine"
        "sine_graph_kuramoto"
        "sine_physics_graph_sine"
        "sine_physics_graph_kuramoto"
    )
    
    declare -a MODEL_NAMES=(
        "Neural ODE (External Noise)"
        "Physics-Informed ODE (External Noise)"
        "Stochastic Neural ODE (Internal Noise)"
        "Oscillator Neural ODE (External Noise)"
        "Oscillator Neural ODE (Internal Noise)"
        "Graph Neural ODE (GCN, External Noise)"
        "Graph Neural ODE (GAT, External Noise)"
        "Physics Graph Neural ODE (GCN, External Noise)"
        "Physics Graph Neural ODE (GAT, External Noise)"
        "Sine Graph Neural ODE (Sine Aggregation, External Noise)"
        "Sine Graph Neural ODE (Kuramoto Aggregation, External Noise)"
        "Sine Physics Graph Neural ODE (Sine Aggregation, External Noise)"
        "Sine Physics Graph Neural ODE (Kuramoto Aggregation, External Noise)"
    )
    
    # Training counters
    local total_trained=0
    local total_failed=0
    local start_time=$(date +%s)
    
    # Train multivariate models
    if [[ "$TRAIN_MULTIVARIATE" == "true" ]]; then
        print_header "TRAINING MULTIVARIATE MODELS (All Climate Variables)"
        
        for i in "${!MODEL_TYPES[@]}"; do
            local model_type="${MODEL_TYPES[$i]}"
            local model_name="${MODEL_NAMES[$i]} - Multivariate"
            
            if train_model "$model_type" "false" "$model_name"; then
                total_trained=$((total_trained + 1))
            else
                total_failed=$((total_failed + 1))
            fi
        done
    fi
    
    # Train ENSO-only models
    if [[ "$TRAIN_ENSO_ONLY" == "true" ]]; then
        print_header "TRAINING ENSO-ONLY MODELS (ENSO Variables Only)"
        
        for i in "${!MODEL_TYPES[@]}"; do
            local model_type="${MODEL_TYPES[$i]}"
            local model_name="${MODEL_NAMES[$i]} - ENSO Only"
            
            if train_model "$model_type" "true" "$model_name"; then
                total_trained=$((total_trained + 1))
            else
                total_failed=$((total_failed + 1))
            fi
        done
    fi
    
    # Final summary
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local duration_minutes=$((duration / 60))
    local duration_seconds=$((duration % 60))
    
    print_header "TRAINING COMPLETE - SUMMARY"
    print_success "Successfully trained: $total_trained models"
    if [[ $total_failed -gt 0 ]]; then
        print_error "Failed to train: $total_failed models"
    fi
    print_info "Total training time: ${duration_minutes}m ${duration_seconds}s"
    print_info "Checkpoints saved in: ./checkpoints/"
    
    # List generated checkpoints
    print_section "Generated Checkpoints"
    if ls checkpoints/*.pt 1> /dev/null 2>&1; then
        ls -la checkpoints/*.pt | while read -r line; do
            print_info "$line"
        done
    else
        print_warning "No checkpoint files found"
    fi
    
    print_header "READY FOR EVALUATION"
    print_info "Run model comparison: python trained_model_comparison.py"
    print_info "Evaluate specific model: python evaluate_neural_ode.py --checkpoint checkpoints/[model].pt --model_type [type]"
    
    if [[ $total_failed -eq 0 ]]; then
        print_success "🎉 ALL MODELS TRAINED SUCCESSFULLY! 🎉"
        exit 0
    else
        print_error "⚠️  SOME MODELS FAILED TO TRAIN ⚠️"
        exit 1
    fi
}

# Run main function
main "$@"
