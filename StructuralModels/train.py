import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import os

from models import DeepMLP, MolecularFeatureExtractor, TraditionalMLEnsemble
from datasets import load_dataset
from config_manager import ConfigManager

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_traditional_ml_models(config_manager: ConfigManager):
    """
    Train traditional ML models on molecular features
    """

    
    # Load dataset
    dataset, train_dataset, valid_dataset, test_dataset = load_dataset(config_manager)
    
    # Extract features
    feature_extractor = MolecularFeatureExtractor()
    
    print("Extracting features for training set...")
    X_train = feature_extractor.fit_transform(train_dataset)
    y_train = np.array([data.y.item() for data in train_dataset])
    
    print("Extracting features for validation set...")
    X_valid = feature_extractor.transform(valid_dataset)
    y_valid = np.array([data.y.item() for data in valid_dataset])
    
    print("Extracting features for test set...")
    X_test = feature_extractor.transform(test_dataset)
    y_test = np.array([data.y.item() for data in test_dataset])
    
    print(f"Feature dimensions: {X_train.shape[1]}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_valid.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # Train ensemble with configuration
    ml_config = config_manager.get_traditional_ml_config()
    ensemble = TraditionalMLEnsemble(config=ml_config)
    ensemble.fit(X_train, y_train)
    
    # Evaluate on validation set
    valid_auc, valid_individual = ensemble.evaluate(X_valid, y_valid)
    print(f"\nValidation Results:")
    print(f"Ensemble AUC: {valid_auc:.4f}")
    for name, score in valid_individual.items():
        print(f"{name}: {score:.4f}")
    
    # Evaluate on test set
    test_auc, test_individual = ensemble.evaluate(X_test, y_test)
    print(f"\nTest Results:")
    print(f"Ensemble AUC: {test_auc:.4f}")
    for name, score in test_individual.items():
        print(f"{name}: {score:.4f}")
    
    # Plot feature importance for tree-based models
    plt.figure(figsize=(15, 8))
    
    # Plot for models that have feature importance
    models_with_importance = {k: v for k, v in ensemble.feature_importance.items() 
                            if len(v) > 0}
    
    if models_with_importance:
        n_models = len(models_with_importance)
        fig, axes = plt.subplots(1, min(n_models, 3), figsize=(15, 5))
        if n_models == 1:
            axes = [axes]
        
        for idx, (name, importance) in enumerate(list(models_with_importance.items())[:3]):
            if idx < len(axes):
                
                # Show top 20 features
                top_indices = np.argsort(importance)[-20:]
                axes[idx].barh(range(20), importance[top_indices])
                axes[idx].set_title(f'Top Features - {name.replace("_", " ").title()}')
                axes[idx].set_xlabel('Feature Importance')
                axes[idx].set_ylabel('Feature Index')
        
        plt.tight_layout()
        plot_path = config_manager.get_plot_path('traditional_ml_feature_importance')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    # Save the trained models
    ensemble_path = config_manager.get_model_path('traditional_ml_ensemble')
    ensemble.save(ensemble_path)
    
    extractor_path = config_manager.get_model_path('traditional_ml_feature_extractor')
    feature_extractor.save(extractor_path)
    
    return ensemble, feature_extractor, test_auc


def train_deep_mlp(config_manager: ConfigManager, epochs=None):
    """
    Train deep MLP on molecular features
    """

    # Get MLP configuration
    mlp_config = config_manager.get_training_config('mlp')
    epochs = epochs or mlp_config.get('epochs', 50)
    batch_size = mlp_config.get('batch_size', 256)
    lr = mlp_config.get('learning_rate', 0.001)
    weight_decay = float(mlp_config.get('weight_decay', 1e-5))
    hidden_dims = mlp_config.get('hidden_dims', [512, 256, 128, 64])
    dropout = mlp_config.get('dropout', 0.3)
    patience = mlp_config.get('patience', 10)
    lr_factor = mlp_config.get('lr_reduction_factor', 0.5)
    
    # Load dataset
    dataset, train_dataset, valid_dataset, test_dataset = load_dataset(config_manager)
    
    # Extract features
    feature_extractor = MolecularFeatureExtractor()
    
    print("Extracting features for MLP...")
    X_train = feature_extractor.fit_transform(train_dataset)
    y_train = np.array([data.y.item() for data in train_dataset])
    
    X_valid = feature_extractor.transform(valid_dataset)
    y_valid = np.array([data.y.item() for data in valid_dataset])
    
    X_test = feature_extractor.transform(test_dataset)
    y_test = np.array([data.y.item() for data in test_dataset])
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_valid_tensor = torch.FloatTensor(X_valid)
    y_valid_tensor = torch.FloatTensor(y_valid).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create data loaders
    train_dataset_mlp = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset_mlp = torch.utils.data.TensorDataset(X_valid_tensor, y_valid_tensor)
    test_dataset_mlp = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = torch.utils.data.DataLoader(train_dataset_mlp, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset_mlp, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset_mlp, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = DeepMLP(
        input_dim=input_dim,
        num_classes=1,
        hidden_dims=hidden_dims,
        dropout=dropout
    ).to(device)
    
    print(f"MLP Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=lr_factor, patience=patience, verbose=True
    )
    
    # Training
    train_losses = []
    valid_aucs = []
    test_aucs = []
    best_valid_auc = 0
    best_test_auc = 0
    
    print("Training Deep MLP...")
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            
            # Validation
            valid_preds = []
            valid_targets = []
            for batch_x, batch_y in valid_loader:
                batch_x = batch_x.to(device)
                out = model(batch_x)
                valid_preds.append(torch.sigmoid(out).cpu())
                valid_targets.append(batch_y.cpu())
            
            valid_preds = torch.cat(valid_preds).numpy()
            valid_targets = torch.cat(valid_targets).numpy()
            valid_auc = roc_auc_score(valid_targets, valid_preds)
            valid_aucs.append(valid_auc)
            
            # Test
            test_preds = []
            test_targets = []
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                out = model(batch_x)
                test_preds.append(torch.sigmoid(out).cpu())
                test_targets.append(batch_y.cpu())
            
            test_preds = torch.cat(test_preds).numpy()
            test_targets = torch.cat(test_targets).numpy()
            test_auc = roc_auc_score(test_targets, test_preds)
            test_aucs.append(test_auc)
        
        scheduler.step(valid_auc)
        
        # Save best model
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_test_auc = test_auc
            model_path = config_manager.get_model_path('best_mlp_model')
            torch.save(model.state_dict(), model_path)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Valid AUC={valid_auc:.4f}, Test AUC={test_auc:.4f}")
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('MLP Training Loss')
    ax1.grid(True)
    
    ax2.plot(valid_aucs, label='Validation AUC')
    ax2.plot(test_aucs, label='Test AUC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('ROC-AUC')
    ax2.set_title('MLP Performance')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = config_manager.get_plot_path('mlp_training_history')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Best MLP Test AUC: {best_test_auc:.4f}")
    print(f"Model saved to: {config_manager.get_model_path('best_mlp_model')}")
    
    # Save the feature extractor as well
    extractor_path = config_manager.get_model_path('mlp_feature_extractor')
    feature_extractor.save(extractor_path)
    
    return model, feature_extractor, best_test_auc


def compare_all_models(config_manager: ConfigManager, quick_mode=False):
    """
    Compare Traditional ML vs Deep MLP models
    """
    
    print("="*60)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*60)
    
    results = {}
    
    # Get epochs for comparison
    if quick_mode:
        comparison_epochs = config_manager.get_comparison_epochs()
        mlp_epochs = comparison_epochs.get('mlp', 10)
    else:
        mlp_config = config_manager.get_training_config('mlp')
        mlp_epochs = mlp_config.get('epochs', 50)
    
    # Train Traditional ML
    print("\n1. Training Traditional ML Ensemble...")
    try:
        ml_ensemble, ml_extractor, ml_auc = train_traditional_ml_models(config_manager)
        results['Traditional ML Ensemble'] = ml_auc
    except Exception as e:
        print(f"Traditional ML training failed: {e}")
        results['Traditional ML Ensemble'] = 0.0
    
    # Train Deep MLP
    print("\n2. Training Deep MLP...")
    try:
        mlp_model, mlp_extractor, mlp_auc = train_deep_mlp(config_manager, epochs=mlp_epochs)
        results['Deep MLP'] = mlp_auc
    except Exception as e:
        print(f"MLP training failed: {e}")
        results['Deep MLP'] = 0.0
    
    # Summary comparison
    print("\n" + "="*60)
    print("FINAL RESULTS COMPARISON")
    print("="*60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (model_name, auc_score) in enumerate(sorted_results, 1):
        print(f"{rank}. {model_name}: {auc_score:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    auc_scores = list(results.values())
    
    colors = config_manager.config.get('visualization', {}).get('color_palette', 
                                                               ['#ff7f0e', '#2ca02c'])
    bars = plt.bar(model_names, auc_scores, color=colors[:len(model_names)])
    plt.ylabel('ROC-AUC Score')
    plt.title('Model Performance Comparison on HIV Dataset\n(Traditional ML vs Deep MLP)')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars, auc_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{score:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plot_path = config_manager.get_plot_path('model_comparison')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nComparison plot saved to: {plot_path}")
    
    return results
