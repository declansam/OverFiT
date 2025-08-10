import yaml
import os
from pathlib import Path
import argparse
from typing import Dict, Any, Optional


class ConfigManager:
    """
    Manages configuration loading and CLI argument overrides
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._create_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        """

        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            print(f"Configuration loaded from {self.config_path}")
            return config

        except FileNotFoundError:
            print(f"Config file {self.config_path} not found. Using default settings.")
            return self._get_default_config()

        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}. Using default settings.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Return default configuration if YAML file is not available
        """

        return {
            'general': {'random_seed': 42, 'device': 'auto'},
            'paths': {'data_dir': 'datasets', 'models_dir': 'models', 'plots_dir': 'plots'},
            'training': {
                'mlp': {'epochs': 50, 'batch_size': 256, 'learning_rate': 0.001, 'weight_decay': 1e-5}
            }
        }
    
    def _create_directories(self):
        """
        Create necessary directories if they don't exist
        """

        paths = self.config.get('paths', {})
        for dir_key, dir_path in paths.items():
            if dir_key.endswith('_dir'):
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                print(f"Directory ensured: {dir_path}")
    
    def get_model_path(self, model_name: str) -> str:
        """
        Get full path for model file
        """

        models_dir = self.config['paths']['models_dir']
        return os.path.join(models_dir, f"{model_name}.pth")
    
    def get_plot_path(self, plot_name: str) -> str:
        """
        Get full path for plot file
        """
        plots_dir = self.config['paths']['plots_dir']
        return os.path.join(plots_dir, f"{plot_name}.png")
    
    def get_training_config(self, model_type: str) -> Dict[str, Any]:
        """
        Get training configuration for specific model type
        """
        return self.config['training'].get(model_type, {})
    
    def get_traditional_ml_config(self) -> Dict[str, Any]:
        """
        Get traditional ML configuration
        """
        return self.config.get('traditional_ml', {})
    
    def update_from_args(self, args: argparse.Namespace):
        """
        Update configuration with CLI arguments
        """

        # Update MLP parameters
        if hasattr(args, 'epochs') and args.epochs is not None:
            self.config['training']['mlp']['epochs'] = args.epochs
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            self.config['training']['mlp']['batch_size'] = args.batch_size
        if hasattr(args, 'lr') and args.lr is not None:
            self.config['training']['mlp']['learning_rate'] = args.lr
        if hasattr(args, 'weight_decay') and args.weight_decay is not None:
            self.config['training']['mlp']['weight_decay'] = args.weight_decay
    
    def get_comparison_epochs(self) -> Dict[str, int]:
        """
        Get epochs for quick comparison mode
        """

        return self.config.get('comparison', {}).get('quick_epochs', {
            'mlp': 10, 'traditional_ml': None
        })
    
    def save_config(self, filepath: Optional[str] = None):
        """
        Save current configuration to file
        """

        save_path = filepath or self.config_path
        with open(save_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)
        print(f"Configuration saved to {save_path}")


def create_argument_parser(config_manager: ConfigManager) -> argparse.ArgumentParser:
    """
    Create argument parser with defaults from config
    """

    parser = argparse.ArgumentParser(description='Molecular Property Prediction with Traditional ML and Deep Learning')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='compare', 
                       choices=['compare', 'train_ml', 'train_mlp', 'predict'],
                       help='Mode to run: compare models, train traditional ML, train MLP, or predict')
    
    # Training parameters (will override config defaults)
    mlp_config = config_manager.get_training_config('mlp')
    parser.add_argument('--epochs', type=int, default=None,
                       help=f'Number of training epochs (default from config: {mlp_config.get("epochs", 50)})')
    parser.add_argument('--batch_size', type=int, default=None,
                       help=f'Training batch size (default from config: {mlp_config.get("batch_size", 256)})')
    parser.add_argument('--lr', type=float, default=None,
                       help=f'Learning rate (default from config: {mlp_config.get("learning_rate", 0.001)})')
    parser.add_argument('--weight_decay', type=float, default=None,
                       help=f'Weight decay (default from config: {mlp_config.get("weight_decay", 0.00001)})')
    
    # Configuration file
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    # Quick mode for comparison
    parser.add_argument('--quick', action='store_true',
                       help='Use reduced epochs for quick comparison')
    
    return parser
