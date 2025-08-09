"""
Main execution script for GNN molecular property prediction

Usage:
    python main.py --mode train --epochs 50 --dataset-name ogbg-molhiv --model-type hybrid
"""

import torch
from config_manager import get_config
from models import get_model
from datasets import load_dataset, create_data_loaders
from train import train_model
from predict import demo_prediction


def main():
    """
    Main function to run the GNN training and prediction pipeline
    """
    
    # Get configuration from YAML and command-line arguments
    config, args = get_config()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Print current configuration
    print("\nCurrent Configuration:")
    config.print_config()
    
    # Load dataset
    dataset, train_dataset, valid_dataset, test_dataset = load_dataset(config)
    
    # Create data loaders
    train_loader, valid_loader, test_loader = create_data_loaders(
        train_dataset, valid_dataset, test_dataset, config
    )
    
    # Initialize model
    model = get_model(
        config=config,
        num_features=dataset.num_features,
        num_classes=dataset.num_tasks,
        device=device
    )
    
    # Train the model
    trained_model, test_auc = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        device=device,
        config=config
    )
    print("Done...")

def run_prediction_only():
    """
    Run only prediction demo without training
    """

    # Get configuration from YAML and command-line arguments
    config, args = get_config()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    demo_prediction(config, device)


if __name__ == "__main__":
    
    # Get configuration to check mode
    config, args = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check if mode is specified in args, otherwise use 'both' as default
    if hasattr(args, 'mode') and args.mode:
        mode = args.mode
    else:
        mode = 'both'
    
    print(f"Running in {mode} mode...")
    
    # Only train
    if mode == 'train':
        print("Running training pipeline...")
        main()

    # Only predict
    elif mode == 'demo':
        print("Running prediction demo...")
        run_prediction_only()

    elif mode == 'both':
        print("Running training + prediction demo...")
        main()
        demo_prediction(config, device)
