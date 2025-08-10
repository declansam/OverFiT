"""
Multi-Layer Perceptron (MLP) model for binary classification on molecular data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for binary classification.
    """
    
    def __init__(self, 
                input_dim: int,
                hidden_dims: List[int] = [512, 256, 128],
                dropout_rate: float = 0.2,
                activation: str = 'relu',
                batch_norm: bool = True):
        """
        Initialize the MLP model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            activation: Activation function ('relu', 'leaky_relu', 'elu', 'gelu')
            batch_norm: Whether to use batch normalization
        """
        super(MLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        
        # Define activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Build layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        self.dropouts = nn.ModuleList()
        
        # Input layer
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            # Linear layer
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
            # Dropout
            self.dropouts.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer (binary classification)
        self.output_layer = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier/He initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)
        
        # Initialize output layer
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output logits of shape (batch_size, 1)
        """
        # Pass through hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Apply batch normalization if enabled
            if self.batch_norm:
                x = self.batch_norms[i](x)
            
            # Apply activation
            x = self.activation(x)
            
            # Apply dropout
            x = self.dropouts[i](x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Probabilities of shape (batch_size, 1)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Get binary predictions.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            threshold: Classification threshold
            
        Returns:
            Binary predictions of shape (batch_size, 1)
        """
        probs = self.predict_proba(x)
        return (probs >= threshold).long()


class MLPEnsemble(nn.Module):
    """
    Ensemble of MLP models for improved robustness.
    """
    
    def __init__(self, 
                 input_dim: int,
                 n_models: int = 5,
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout_rate: float = 0.2,
                 activation: str = 'relu',
                 batch_norm: bool = True):
        """
        Initialize the MLP ensemble.
        
        Args:
            input_dim: Input feature dimension
            n_models: Number of models in ensemble
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            activation: Activation function
            batch_norm: Whether to use batch normalization
        """
        super(MLPEnsemble, self).__init__()
        
        self.n_models = n_models
        self.models = nn.ModuleList([
            MLP(input_dim, hidden_dims, dropout_rate, activation, batch_norm)
            for _ in range(n_models)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble (returns average predictions).
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Average logits of shape (batch_size, 1)
        """
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Average the logits
        return torch.mean(torch.stack(outputs), dim=0)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get ensemble prediction probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Average probabilities of shape (batch_size, 1)
        """
        with torch.no_grad():
            probs = []
            for model in self.models:
                model_probs = model.predict_proba(x)
                probs.append(model_probs)
            
            # Average the probabilities
            return torch.mean(torch.stack(probs), dim=0)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Get ensemble binary predictions.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            threshold: Classification threshold
            
        Returns:
            Binary predictions of shape (batch_size, 1)
        """
        probs = self.predict_proba(x)
        return (probs >= threshold).long()


def create_mlp_model(input_dim: int, 
                     model_type: str = 'standard',
                     hidden_dims: Optional[List[int]] = None,
                     dropout_rate: float = 0.2,
                     activation: str = 'relu',
                     batch_norm: bool = True,
                     n_ensemble: int = 5) -> nn.Module:
    """
    Factory function to create MLP models.
    
    Args:
        input_dim: Input feature dimension
        model_type: Type of model ('standard', 'ensemble', 'large', 'small')
        hidden_dims: Custom hidden dimensions (overrides model_type defaults)
        dropout_rate: Dropout probability
        activation: Activation function
        batch_norm: Whether to use batch normalization
        n_ensemble: Number of models for ensemble
        
    Returns:
        MLP model instance
    """
    # Default architectures
    if hidden_dims is None:
        if model_type == 'small':
            hidden_dims = [128, 64]
        elif model_type == 'large':
            hidden_dims = [1024, 512, 256, 128]
        else:  # standard
            hidden_dims = [512, 256, 128]
    
    if model_type == 'ensemble':
        return MLPEnsemble(
            input_dim=input_dim,
            n_models=n_ensemble,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm
        )
    else:
        return MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm
        )


if __name__ == "__main__":
    # Test the model
    input_dim = 2048 + 24  # Morgan fingerprint + descriptors
    
    # Test standard MLP
    model = create_mlp_model(input_dim, model_type='standard')
    print(f"Standard MLP: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test ensemble
    ensemble_model = create_mlp_model(input_dim, model_type='ensemble')
    print(f"Ensemble MLP: {sum(p.numel() for p in ensemble_model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, input_dim)
    
    with torch.no_grad():
        output = model(x)
        probs = model.predict_proba(x)
        preds = model.predict(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Probabilities shape: {probs.shape}")
        print(f"Predictions shape: {preds.shape}")
        print(f"Probability range: [{probs.min():.3f}, {probs.max():.3f}]")
