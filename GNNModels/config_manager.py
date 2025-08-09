"""
Configuration management for GNN molecular property prediction
"""

import yaml
import argparse
import os
from typing import Dict, Any


class Config:
    """Configuration class to handle YAML config files and command-line arguments"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration from YAML file
        
        Args:
            config_path (str): Path to the YAML configuration file
        """

        self.config_path = config_path
        self.config = self._load_yaml_config()
    
    def _load_yaml_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        """

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file {self.config_path} not found!")
        
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        return config
    
    def _update_model_dependent_paths(self):
        """
        Update file paths based on model type
        """
        model_type = self.config.get('model', {}).get('type', 'hybrid')
        
        # Update model save path
        self.config['prediction']['model_save_path'] = f"models/best_hiv_{model_type}_model.pth"
        
        # Update plot path
        self.config['visualization']['plot_path'] = f"plots/training_history_{model_type}.png"
    
    def update_from_args(self, args: argparse.Namespace) -> None:
        """
        Update configuration with command-line arguments
        
        Args:
            args (argparse.Namespace): Parsed command-line arguments
        """

        # Dataset configuration
        if hasattr(args, 'dataset_name') and args.dataset_name is not None:
            self.config['dataset']['name'] = args.dataset_name
        if hasattr(args, 'dataset_root') and args.dataset_root is not None:
            self.config['dataset']['root'] = args.dataset_root
        
        # Model configuration
        if hasattr(args, 'model_type') and args.model_type is not None:
            self.config['model']['type'] = args.model_type
        if hasattr(args, 'hidden_dim') and args.hidden_dim is not None:
            self.config['model']['hidden_dim'] = args.hidden_dim
        if hasattr(args, 'dropout') and args.dropout is not None:
            self.config['model']['dropout'] = args.dropout
        if hasattr(args, 'num_layers') and args.num_layers is not None:
            self.config['model']['num_layers'] = args.num_layers
        
        # Training configuration
        if hasattr(args, 'epochs') and args.epochs is not None:
            self.config['training']['epochs'] = args.epochs
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            self.config['training']['batch_size'] = args.batch_size
        if hasattr(args, 'lr') and args.lr is not None:
            self.config['training']['lr'] = args.lr
        if hasattr(args, 'weight_decay') and args.weight_decay is not None:
            self.config['training']['weight_decay'] = args.weight_decay
        
        # Prediction configuration
        if hasattr(args, 'model_save_path') and args.model_save_path is not None:
            self.config['prediction']['model_save_path'] = args.model_save_path
        if hasattr(args, 'threshold') and args.threshold is not None:
            self.config['prediction']['threshold'] = args.threshold
        
        # Update model-dependent paths after all arguments are processed
        self._update_model_dependent_paths()
    
    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation
        
        Args:
            key_path (str): Dot-separated path to the configuration value (e.g., 'model.type')
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def save(self, path: str = None) -> None:
        """
        Save current configuration to YAML file
        
        Args:
            path (str): Path to save the configuration file. If None, uses original path.
        """
        save_path = path or self.config_path
        with open(save_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)
        print(f"Configuration saved to {save_path}")
    
    def print_config(self) -> None:
        """Print current configuration"""
        print("Current Configuration:")
        print(yaml.dump(self.config, default_flow_style=False, indent=2))


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='GNN Molecular Property Prediction')
    
    # General arguments
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--mode', type=str, choices=['train', 'demo', 'both'], default='both',
                       help='Execution mode: train, predict, or both (default: both)')
    
    # Dataset arguments
    parser.add_argument('--dataset-name', type=str,
                       help='Dataset name (default: from config)')
    parser.add_argument('--dataset-root', type=str,
                       help='Dataset root directory (default: from config)')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, choices=['gin', 'hybrid'],
                       help='Model architecture type (default: from config)')
    parser.add_argument('--hidden-dim', type=int,
                       help='Hidden dimension size (default: from config)')
    parser.add_argument('--dropout', type=float,
                       help='Dropout rate (default: from config)')
    parser.add_argument('--num-layers', type=int,
                       help='Number of layers for GIN model (default: from config)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int,
                       help='Number of training epochs (default: from config)')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size (default: from config)')
    parser.add_argument('--lr', type=float,
                       help='Learning rate (default: from config)')
    parser.add_argument('--weight-decay', type=float,
                       help='Weight decay (default: from config)')
    
    # Prediction arguments
    parser.add_argument('--model-save-path', type=str,
                       help='Path to save/load model (default: from config)')
    parser.add_argument('--threshold', type=float,
                       help='Prediction threshold (default: from config)')
    
    # Utility arguments
    parser.add_argument('--print-config', action='store_true',
                       help='Print current configuration and exit')
    parser.add_argument('--save-config', type=str,
                       help='Save current configuration to specified path')
    
    return parser.parse_args()


def get_config() -> Config:
    """
    Get configuration from YAML file and command-line arguments
    
    Returns:
        Config: Configuration object
    """
    args = parse_arguments()
    
    # Load configuration from YAML
    config = Config(args.config)
    
    # Update with command-line arguments
    config.update_from_args(args)
    
    # Handle utility arguments
    if args.print_config:
        config.print_config()
        exit(0)
    
    if args.save_config:
        config.save(args.save_config)
        exit(0)
    
    return config, args
