#!/usr/bin/env python3
"""
Modular ML Implementation for Molecular Property Prediction

This modular implementation provides approaches for molecular property prediction:
- Traditional Machine Learning (Random Forest, XGBoost, etc.)
- Deep MLP on molecular features

Usage:
    python main.py [options]
    python main.py --mode [compare, train_ml, train_mlp, predict] --epochs 50

Modules:
    - models.py: Neural network architectures and ML models
    - datasets.py: Data loading and preprocessing utilities
    - train.py: Training functions for all model types
    - predict.py: Prediction and inference utilities
    - config_manager.py: Configuration management and CLI parsing
    - config.yaml: Configuration file with default parameters
"""

import sys
import warnings
import torch
import numpy as np
warnings.filterwarnings('ignore')

from config_manager import ConfigManager, create_argument_parser
from train import train_traditional_ml_models, train_deep_mlp, compare_all_models
from predict import demo_prediction

# Set random seeds for reproducibility
def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    
    # Initialize configuration manager
    config_manager = ConfigManager()
    
    # Create argument parser with config defaults
    parser = create_argument_parser(config_manager)
    args = parser.parse_args()
    
    # Update config with custom config file if provided
    if args.config != 'config.yaml':
        config_manager = ConfigManager(args.config)
    
    # Update configuration with CLI arguments
    config_manager.update_from_args(args)
    
    # Set random seeds
    seed = config_manager.config.get('general', {}).get('random_seed', 42)
    set_seeds(seed)
    
    print("="*60)
    print("MOLECULAR PROPERTY PREDICTION")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Configuration loaded from: {config_manager.config_path}")
    if args.quick:
        print("Quick mode enabled - using reduced epochs")
    
    if args.mode == 'compare':
        print("Running comprehensive model comparison...")
        results = compare_all_models(config_manager, quick_mode=args.quick)
        
    elif args.mode == 'train_ml':
        print("Training Traditional ML Ensemble...")
        ensemble, feature_extractor, ml_auc = train_traditional_ml_models(config_manager)
        print(f"Final Test AUC: {ml_auc:.4f}")
        
    elif args.mode == 'train_mlp':
        print("Training Deep MLP...")
        mlp_model, mlp_extractor, mlp_auc = train_deep_mlp(config_manager, epochs=args.epochs)
        print(f"Final Test AUC: {mlp_auc:.4f}")
        
    elif args.mode == 'predict':
        print("Running prediction demo...")
        demo_prediction(config_manager)
    
    print("\nExecution completed!")


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print("No arguments provided. Running default comparison...")
        
        # Initialize config manager and run comparison
        config_manager = ConfigManager()
        set_seeds(config_manager.config.get('general', {}).get('random_seed', 42))
        
        print("Starting comprehensive model comparison...")
        results = compare_all_models(config_manager, quick_mode=True)
        
        # Demo predictions
        demo_prediction(config_manager)
    else:
        main()