"""
Training and evaluation functionality for GNN models
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from ogb.graphproppred import Evaluator
import matplotlib.pyplot as plt
import os


def train_epoch(model, device, loader, optimizer, criterion):
    """
    Train for one epoch
    """

    # Init
    model.train()
    total_loss = 0

    # Iterate over batches
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y.float())

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, device, loader, evaluator):
    """
    Evaluate model performance
    """

    # Init
    model.eval()
    y_true = []
    y_pred = []

    # Iterate over batches
    for batch in tqdm(loader, desc="Evaluating"):
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)

        y_true.append(batch.y.cpu())
        y_pred.append(out.cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    # Calculate ROC-AUC
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    result = evaluator.eval(input_dict)

    return result['rocauc']


def train_model(model, train_loader, valid_loader, test_loader, device, config):
    """
    Main training function
    """
    
    # Get parameters from config
    epochs = config.get('training.epochs', 100)
    lr = config.get('training.lr', 0.001)
    weight_decay = config.get('training.weight_decay', 0.0)
    patience = config.get('training.patience', 10)
    lr_factor = config.get('training.lr_factor', 0.5)
    save_path = config.get('prediction.model_save_path', 'best_hiv_gnn_model.pth')
    dataset_name = config.get('dataset.name', 'ogbg-molhiv')
    
    print(f"\nModel architecture: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=lr_factor, patience=patience, verbose=True
    )

    # Evaluator
    evaluator = Evaluator(name=dataset_name)

    # Training history
    train_losses = []
    valid_aucs = []
    test_aucs = []
    best_valid_auc = 0
    best_test_auc = 0

    print("\nStarting training...")
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Train
        train_loss = train_epoch(model, device, train_loader, optimizer, criterion)
        train_losses.append(train_loss)

        # Evaluate
        valid_auc = evaluate(model, device, valid_loader, evaluator)
        test_auc = evaluate(model, device, test_loader, evaluator)

        valid_aucs.append(valid_auc)
        test_aucs.append(test_auc)

        # Update learning rate
        scheduler.step(valid_auc)

        # Save best model
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_test_auc = test_auc
            
            # Create models directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

        print(f"Loss: {train_loss:.4f} | Valid AUC: {valid_auc:.4f} | Test AUC: {test_auc:.4f}")
        print(f"Best Valid AUC: {best_valid_auc:.4f} | Best Test AUC: {best_test_auc:.4f}")

    # Plot training history
    plot_training_history(train_losses, valid_aucs, test_aucs, config)

    print(f"\nTraining complete! Best Valid AUC: {best_valid_auc:.4f}, Best Test AUC: {best_test_auc:.4f}")

    return model, best_test_auc


def plot_training_history(train_losses, valid_aucs, test_aucs, config):
    """
    Plot and save training history
    """
    
    save_plots = config.get('visualization.save_plots', True)
    save_path = config.get('visualization.plot_path', 'training_history.png')

    # Subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.grid(True)

    ax2.plot(valid_aucs, label='Validation AUC')
    ax2.plot(test_aucs, label='Test AUC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('ROC-AUC')
    ax2.set_title('Model Performance Over Time')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    
    if save_plots:
        # Create plots directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Training history saved to {save_path}")
    
    plt.show()
