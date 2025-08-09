"""
Training script for MLP classification on ogb-molhiv dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from ogb.graphproppred import Evaluator
import os
import argparse
from typing import Dict, List, Tuple
import time
from tqdm import tqdm

from data_loader import prepare_data_loaders
from model import create_mlp_model

import warnings
warnings.filterwarnings('ignore')

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, loss: float, model: nn.Module) -> bool:
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    }
    return metrics


def evaluate_model(model: nn.Module, 
                  data_loader: DataLoader, 
                  criterion: nn.Module,
                  device: torch.device,
                  evaluator: Evaluator = None) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate model performance on a dataset.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        criterion: Loss function
        device: Device to run evaluation on
        evaluator: OGB evaluator (optional)
        
    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device).float()
            
            # Forward pass
            outputs = model(batch_features).squeeze()
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
            
            # Get predictions and probabilities
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    
    # Add OGB evaluation if available
    if evaluator is not None:
        try:
            ogb_result = evaluator.eval({
                'y_true': all_labels.reshape(-1, 1),
                'y_pred': all_probs.reshape(-1, 1)
            })
            metrics['ogb_rocauc'] = ogb_result['rocauc']
        except:
            pass
    
    return avg_loss, metrics


def train_epoch(model: nn.Module,
               train_loader: DataLoader,
               criterion: nn.Module,
               optimizer: optim.Optimizer,
               device: torch.device,
               scheduler: optim.lr_scheduler._LRScheduler = None) -> float:
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        scheduler: Learning rate scheduler (optional)
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    
    for batch_features, batch_labels in train_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device).float()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_features).squeeze()
        loss = criterion(outputs, batch_labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()
    
    if scheduler is not None:
        scheduler.step()
    
    return total_loss / len(train_loader)


def plot_training_curves(train_losses: List[float],
                        val_losses: List[float],
                        train_metrics: List[Dict[str, float]],
                        val_metrics: List[Dict[str, float]],
                        save_path: str = "mlp/training_curves.png"):
    """
    Plot training curves for loss and metrics.
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Loss curves
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss')
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Metric curves
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    for i, metric in enumerate(metric_names):
        row = (i + 1) // 3
        col = (i + 1) % 3
        
        train_values = [m[metric] for m in train_metrics]
        val_values = [m[metric] for m in val_metrics]
        
        axes[row, col].plot(epochs, train_values, 'b-', label=f'Training {metric.upper()}')
        axes[row, col].plot(epochs, val_values, 'r-', label=f'Validation {metric.upper()}')
        axes[row, col].set_title(f'Training and Validation {metric.upper()}')
        axes[row, col].set_xlabel('Epoch')
        axes[row, col].set_ylabel(metric.upper())
        axes[row, col].legend()
        axes[row, col].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {save_path}")


def train_mlp(config: Dict) -> nn.Module:
    """
    Main training function for MLP model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Trained model
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    print("Loading data...")
    train_loader, val_loader, test_loader, scaler = prepare_data_loaders(
        data_dir=config['data_dir'],
        morgan_radius=config['morgan_radius'],
        morgan_bits=config['morgan_bits'],
        include_descriptors=config['include_descriptors'],
        batch_size=config['batch_size'],
        cache_features=config['cache_features']
    )
    
    # Calculate input dimension
    for batch_features, _ in train_loader:
        input_dim = batch_features.shape[1]
        break
    
    print(f"Input dimension: {input_dim}")
    
    # Create model
    model = create_mlp_model(
        input_dim=input_dim,
        model_type=config['model_type'],
        hidden_dims=config['hidden_dims'],
        dropout_rate=config['dropout_rate'],
        activation=config['activation'],
        batch_norm=config['batch_norm'],
        n_ensemble=config['n_ensemble']
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config['patience'], min_delta=0.001)
    
    # OGB evaluator
    evaluator = Evaluator(name='ogbg-molhiv')
    
    # Training loop
    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []
    
    best_val_rocauc = 0.0
    best_model_state = None
    
    print("Starting training...")
    for epoch in range(config['epochs']):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        val_loss, val_metrics_dict = evaluate_model(model, val_loader, criterion, device, evaluator)
        train_loss_eval, train_metrics_dict = evaluate_model(model, train_loader, criterion, device, evaluator)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_metrics.append(train_metrics_dict)
        val_metrics.append(val_metrics_dict)
        
        # Save best model
        current_val_rocauc = val_metrics_dict.get('ogb_rocauc', val_metrics_dict['roc_auc'])
        if current_val_rocauc > best_val_rocauc:
            best_val_rocauc = current_val_rocauc
            best_model_state = model.state_dict().copy()
        
        epoch_time = time.time() - start_time
        
        # Print progress
        if epoch % config['print_every'] == 0 or epoch == config['epochs'] - 1:
            print(f"Epoch {epoch+1}/{config['epochs']} ({epoch_time:.2f}s)")
            print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"  Train ROC-AUC: {train_metrics_dict.get('ogb_rocauc', train_metrics_dict['roc_auc']):.4f}")
            print(f"  Val ROC-AUC: {current_val_rocauc:.4f}")
            print(f"  Val F1: {val_metrics_dict['f1']:.4f}")
        
        # Early stopping
        if early_stopping(val_loss, model):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    print("\nFinal Evaluation:")
    test_loss, test_metrics = evaluate_model(model, test_loader, criterion, device, evaluator)
    
    print(f"Test Loss: {test_loss:.4f}")
    for metric, value in test_metrics.items():
        print(f"Test {metric.upper()}: {value:.4f}")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_metrics, val_metrics)
    
    # Save model
    model_path = os.path.join("mlp", f"best_mlp_model_{config['model_type']}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'test_metrics': test_metrics,
        'scaler': scaler
    }, model_path)
    
    print(f"\nModel saved to {model_path}")
    
    return model


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Train MLP for molecular classification')
    
    # Data parameters
    parser.add_argument('--data_dir', default='data', help='Data directory')
    parser.add_argument('--morgan_radius', type=int, default=3, help='Morgan fingerprint radius')
    parser.add_argument('--morgan_bits', type=int, default=2048, help='Morgan fingerprint bits')
    parser.add_argument('--include_descriptors', action='store_true', help='Include molecular descriptors')
    parser.add_argument('--cache_features', action='store_true', help='Cache extracted features')
    
    # Model parameters
    parser.add_argument('--model_type', default='standard', choices=['standard', 'small', 'large', 'ensemble'], 
                       help='Model architecture type')
    parser.add_argument('--hidden_dims', nargs='+', type=int, help='Custom hidden dimensions')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--activation', default='relu', choices=['relu', 'leaky_relu', 'elu', 'gelu'], 
                       help='Activation function')
    parser.add_argument('--batch_norm', action='store_true', help='Use batch normalization')
    parser.add_argument('--n_ensemble', type=int, default=5, help='Number of models for ensemble')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--print_every', type=int, default=10, help='Print frequency')
    
    args = parser.parse_args()
    
    # Convert to config dict
    config = vars(args)
    
    # Create output directory
    os.makedirs("mlp", exist_ok=True)
    
    # Set default flags
    config['include_descriptors'] = True
    config['cache_features'] = True
    config['batch_norm'] = True
    
    # Print configuration
    print("Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Train model
    model = train_mlp(config)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
