"""
Evaluation script for trained MLP models on ogb-molhiv dataset.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report,
                           precision_recall_curve, roc_curve)
from ogb.graphproppred import Evaluator
import os
import argparse
from typing import Dict, Tuple, List
import joblib

from data_loader import prepare_data_loaders
from model import create_mlp_model


def load_trained_model(model_path: str, device: torch.device) -> Tuple[nn.Module, Dict, object]:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Tuple of (model, config, scaler)
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    scaler = checkpoint['scaler']
    
    # Reconstruct model
    # We need to get input dimension from scaler or config
    if hasattr(scaler, 'n_features_in_'):
        input_dim = scaler.n_features_in_
    else:
        # Fallback: reconstruct based on config
        input_dim = config['morgan_bits']
        if config['include_descriptors']:
            input_dim += 24  # Number of molecular descriptors
    
    model = create_mlp_model(
        input_dim=input_dim,
        model_type=config['model_type'],
        hidden_dims=config.get('hidden_dims'),
        dropout_rate=config['dropout_rate'],
        activation=config['activation'],
        batch_norm=config['batch_norm'],
        n_ensemble=config.get('n_ensemble', 5)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, config, scaler


def evaluate_comprehensive(model: nn.Module,
                         data_loader,
                         device: torch.device,
                         dataset_name: str = "Test",
                         save_plots: bool = True,
                         output_dir: str = "mlp") -> Dict[str, float]:
    """
    Comprehensive evaluation with detailed metrics and visualizations.
    
    Args:
        model: Trained model
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        dataset_name: Name of dataset being evaluated
        save_plots: Whether to save visualization plots
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    print(f"Evaluating on {dataset_name} set...")
    
    with torch.no_grad():
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            outputs = model(batch_features).squeeze()
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'roc_auc': roc_auc_score(all_labels, all_probs)
    }
    
    # OGB evaluation
    evaluator = Evaluator(name='ogbg-molhiv')
    ogb_result = evaluator.eval({
        'y_true': all_labels.reshape(-1, 1),
        'y_pred': all_probs.reshape(-1, 1)
    })
    metrics['ogb_rocauc'] = ogb_result['rocauc']
    
    # Print metrics
    print(f"\n{dataset_name} Results:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Detailed classification report
    print(f"\n{dataset_name} Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Non-HIV-active', 'HIV-active']))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n{dataset_name} Confusion Matrix:")
    print(cm)
    
    if save_plots:
        # Create visualizations
        create_evaluation_plots(all_labels, all_preds, all_probs, dataset_name, output_dir)
    
    return metrics


def create_evaluation_plots(y_true: np.ndarray,
                          y_pred: np.ndarray, 
                          y_prob: np.ndarray,
                          dataset_name: str,
                          output_dir: str = "mlp"):
    """
    Create comprehensive evaluation plots.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        dataset_name: Name of dataset
        output_dir: Output directory for plots
    """
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title(f'{dataset_name} Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('True')
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title(f'{dataset_name} ROC Curve')
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(True)
    
    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = np.mean(precision)
    axes[0, 2].plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
    axes[0, 2].set_xlabel('Recall')
    axes[0, 2].set_ylabel('Precision')
    axes[0, 2].set_title(f'{dataset_name} Precision-Recall Curve')
    axes[0, 2].legend(loc="lower left")
    axes[0, 2].grid(True)
    
    # 4. Probability Distribution
    axes[1, 0].hist(y_prob[y_true == 0], bins=50, alpha=0.7, label='Non-HIV-active', color='blue')
    axes[1, 0].hist(y_prob[y_true == 1], bins=50, alpha=0.7, label='HIV-active', color='red')
    axes[1, 0].set_xlabel('Predicted Probability')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'{dataset_name} Probability Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 5. Calibration Plot
    from sklearn.calibration import calibration_curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)
    axes[1, 1].plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    axes[1, 1].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    axes[1, 1].set_xlabel('Mean Predicted Probability')
    axes[1, 1].set_ylabel('Fraction of Positives')
    axes[1, 1].set_title(f'{dataset_name} Calibration Plot')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 6. Class Distribution
    class_counts = np.bincount(y_true)
    pred_counts = np.bincount(y_pred)
    x = ['Non-HIV-active', 'HIV-active']
    x_pos = np.arange(len(x))
    
    width = 0.35
    axes[1, 2].bar(x_pos - width/2, class_counts, width, label='True', alpha=0.8)
    axes[1, 2].bar(x_pos + width/2, pred_counts, width, label='Predicted', alpha=0.8)
    axes[1, 2].set_xlabel('Class')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title(f'{dataset_name} Class Distribution')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(x)
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"{dataset_name.lower()}_evaluation_plots.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"{dataset_name} evaluation plots saved to {plot_path}")


def compare_thresholds(model: nn.Module,
                      data_loader,
                      device: torch.device,
                      thresholds: List[float] = None,
                      output_dir: str = "mlp") -> pd.DataFrame:
    """
    Evaluate model performance across different classification thresholds.
    
    Args:
        model: Trained model
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        thresholds: List of thresholds to evaluate
        output_dir: Output directory
        
    Returns:
        DataFrame with metrics for each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)
    
    model.eval()
    
    # Get all predictions and labels
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features).squeeze()
            probs = torch.sigmoid(outputs)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Evaluate at different thresholds
    results = []
    for threshold in thresholds:
        preds = (all_probs >= threshold).astype(int)
        
        metrics = {
            'threshold': threshold,
            'accuracy': accuracy_score(all_labels, preds),
            'precision': precision_score(all_labels, preds, zero_division=0),
            'recall': recall_score(all_labels, preds, zero_division=0),
            'f1': f1_score(all_labels, preds, zero_division=0),
            'specificity': recall_score(1 - all_labels, 1 - preds, zero_division=0)
        }
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    
    # Plot threshold analysis
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(results_df['threshold'], results_df['accuracy'], 'o-', label='Accuracy')
    plt.plot(results_df['threshold'], results_df['f1'], 's-', label='F1-Score')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Accuracy and F1-Score vs Threshold')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(results_df['threshold'], results_df['precision'], 'o-', label='Precision')
    plt.plot(results_df['threshold'], results_df['recall'], 's-', label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision and Recall vs Threshold')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(results_df['threshold'], results_df['recall'], 'o-', label='Sensitivity (Recall)')
    plt.plot(results_df['threshold'], results_df['specificity'], 's-', label='Specificity')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Sensitivity and Specificity vs Threshold')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    # Find optimal threshold (max F1)
    optimal_idx = results_df['f1'].idxmax()
    optimal_threshold = results_df.loc[optimal_idx, 'threshold']
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Optimal (F1) = {optimal_threshold:.2f}')
    plt.plot(results_df['threshold'], results_df['f1'], 'o-', label='F1-Score')
    plt.xlabel('Threshold')
    plt.ylabel('F1-Score')
    plt.title('F1-Score vs Threshold')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot and results
    plot_path = os.path.join(output_dir, "threshold_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    csv_path = os.path.join(output_dir, "threshold_analysis.csv")
    results_df.to_csv(csv_path, index=False)
    
    print(f"Threshold analysis saved to {plot_path} and {csv_path}")
    print(f"Optimal threshold (max F1): {optimal_threshold:.3f} (F1 = {results_df.loc[optimal_idx, 'f1']:.3f})")
    
    return results_df


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained MLP model')
    parser.add_argument('--model_path', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', default='data', help='Data directory')
    parser.add_argument('--output_dir', default='mlp', help='Output directory for plots')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation')
    parser.add_argument('--threshold_analysis', action='store_true', help='Perform threshold analysis')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model, config, scaler = load_trained_model(args.model_path, device)
    
    print("Model configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Prepare data loaders using saved configuration
    _, val_loader, test_loader, _ = prepare_data_loaders(
        data_dir=args.data_dir,
        morgan_radius=config['morgan_radius'],
        morgan_bits=config['morgan_bits'],
        include_descriptors=config['include_descriptors'],
        batch_size=args.batch_size,
        cache_features=config['cache_features']
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate on validation and test sets
    val_metrics = evaluate_comprehensive(model, val_loader, device, "Validation", True, args.output_dir)
    test_metrics = evaluate_comprehensive(model, test_loader, device, "Test", True, args.output_dir)
    
    # Threshold analysis
    if args.threshold_analysis:
        print("\nPerforming threshold analysis...")
        threshold_results = compare_thresholds(model, test_loader, device, output_dir=args.output_dir)
    
    # Save results summary
    results_summary = {
        'model_path': args.model_path,
        'config': config,
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics
    }
    
    summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
    import json
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nEvaluation summary saved to {summary_path}")
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
