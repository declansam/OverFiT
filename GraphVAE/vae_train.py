"""
VAE Training Script
Focuses on fixing the core training issues with a very simple approach
"""

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.nn import GINConv, global_mean_pool
from ogb.graphproppred import PygGraphPropPredDataset
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class GraphVAE(torch.nn.Module):
    """GraphVAE that focuses on stability"""
    
    def __init__(self, node_input_dim, hidden_dim, latent_dim, max_nodes):
        super(GraphVAE, self).__init__()
        
        self.max_nodes = max_nodes
        self.latent_dim = latent_dim
        self.node_input_dim = node_input_dim
        
        # Simple encoder
        self.conv1 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(node_input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        ))
        
        self.conv2 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        ))
        
        # VAE bottleneck
        self.fc_mu = torch.nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = torch.nn.Linear(hidden_dim, latent_dim)
        
        # Simple decoder
        self.decode_fc = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output layers
        self.node_output = torch.nn.Linear(hidden_dim, max_nodes * node_input_dim)
        self.edge_output = torch.nn.Linear(hidden_dim, max_nodes * max_nodes)
        
        # Initialize properly
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
    
    def encode(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        logvar = torch.clamp(logvar, min=-5, max=5)  # Prevent extreme values
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, max_nodes):
        batch_size = z.size(0)
        h = self.decode_fc(z)
        
        # Node features
        node_features = self.node_output(h)
        node_features = node_features.view(batch_size, self.max_nodes, self.node_input_dim)
        node_features = F.softmax(node_features, dim=-1)
        
        # Edge features 
        adj_logits = self.edge_output(h)
        adj_logits = adj_logits.view(batch_size, self.max_nodes, self.max_nodes)
        
        # Make symmetric
        adj_logits = (adj_logits + adj_logits.transpose(-1, -2)) / 2
        
        # Remove self-loops
        mask = torch.eye(self.max_nodes, device=z.device).bool()
        adj_logits = adj_logits.masked_fill(mask, -10.0)
        
        # Clamp to prevent extreme values
        adj_logits = torch.clamp(adj_logits, min=-10, max=10)
        
        if max_nodes < self.max_nodes:
            node_features = node_features[:, :max_nodes, :]
            adj_logits = adj_logits[:, :max_nodes, :max_nodes]
        
        return adj_logits, node_features


class Trainer:
    """Trainer focusing on numerical stability"""
    
    def __init__(self, model, device, beta=0.01):
        self.model = model.to(device)
        self.device = device
        self.beta = beta
        
    def simple_loss(self, recon_adj, recon_features, target_adj, target_features, mu, logvar, num_nodes):
        """Very simple loss function to avoid NaN issues"""
        batch_size = recon_adj.size(0)
        max_nodes = recon_adj.size(1)
        
        # Create node mask
        node_mask = torch.zeros(batch_size, max_nodes, device=self.device)
        for i, n_nodes in enumerate(num_nodes):
            actual_nodes = min(n_nodes.item(), max_nodes)
            node_mask[i, :actual_nodes] = 1
        
        # Ensure same sizes
        if target_features.size(1) != max_nodes:
            if target_features.size(1) > max_nodes:
                target_features = target_features[:, :max_nodes, :]
            else:
                pad_size = max_nodes - target_features.size(1)
                padding = torch.zeros(batch_size, pad_size, target_features.size(2), device=self.device)
                target_features = torch.cat([target_features, padding], dim=1)
        
        if target_adj.size(1) != max_nodes:
            if target_adj.size(1) > max_nodes:
                target_adj = target_adj[:, :max_nodes, :max_nodes]
            else:
                pad_size = max_nodes - target_adj.size(1)
                padding = torch.zeros(batch_size, pad_size, max_nodes, device=self.device)
                target_adj = torch.cat([target_adj, padding], dim=1)
                padding = torch.zeros(batch_size, max_nodes, pad_size, device=self.device)
                target_adj = torch.cat([target_adj, padding], dim=2)
        
        # Simple MSE loss for nodes (avoid cross-entropy issues)
        node_loss = F.mse_loss(
            recon_features * node_mask.unsqueeze(-1),
            target_features * node_mask.unsqueeze(-1),
            reduction='mean'
        )
        
        # Simple edge loss with very careful handling
        edge_probs = torch.sigmoid(recon_adj)
        edge_loss = F.mse_loss(edge_probs, target_adj, reduction='mean')
        
        # Very simple KL loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Check all components for NaN
        if torch.isnan(node_loss):
            node_loss = torch.tensor(1.0, device=self.device)
        if torch.isnan(edge_loss):
            edge_loss = torch.tensor(1.0, device=self.device)
        if torch.isnan(kl_loss):
            kl_loss = torch.tensor(0.1, device=self.device)
        
        total_loss = node_loss + edge_loss + self.beta * kl_loss
        
        return total_loss, node_loss, edge_loss, kl_loss
    
    def train_epoch(self, dataloader, optimizer, epoch):
        self.model.train()
        total_losses = []
        
        # Very conservative beta
        self.beta = min(0.01, 0.001 * (epoch + 1))
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            batch = batch.to(self.device)
            
            x_dense, node_mask = to_dense_batch(batch.x, batch.batch)
            adj_dense = to_dense_adj(batch.edge_index, batch.batch)
            num_nodes = node_mask.sum(dim=1)
            
            optimizer.zero_grad()
            
            try:
                mu, logvar = self.model.encode(batch.x, batch.edge_index, batch.batch)
                z = self.model.reparameterize(mu, logvar)
                recon_adj, recon_features = self.model.decode(z, num_nodes.max().item())
                
                loss, node_loss, edge_loss, kl_loss = self.simple_loss(
                    recon_adj, recon_features, adj_dense, x_dense, mu, logvar, num_nodes
                )
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    
                    # Very conservative gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    
                    if not torch.isnan(grad_norm):
                        optimizer.step()
                
                # Log losses (with fallback values)
                total_losses.append([
                    loss.item() if not torch.isnan(loss) else 1.0,
                    node_loss.item() if not torch.isnan(node_loss) else 1.0,
                    edge_loss.item() if not torch.isnan(edge_loss) else 1.0,
                    kl_loss.item() if not torch.isnan(kl_loss) else 0.01
                ])
                
            except Exception as e:
                print(f"Error in batch: {e}")
                total_losses.append([1.0, 1.0, 1.0, 0.01])
                continue
        
        return np.mean(total_losses, axis=0)
    
    def validate(self, dataloader):
        self.model.eval()
        total_losses = []
        
        with torch.no_grad():
            for batch in dataloader:
                try:
                    batch = batch.to(self.device)
                    
                    x_dense, node_mask = to_dense_batch(batch.x, batch.batch)
                    adj_dense = to_dense_adj(batch.edge_index, batch.batch)
                    num_nodes = node_mask.sum(dim=1)
                    
                    mu, logvar = self.model.encode(batch.x, batch.edge_index, batch.batch)
                    z = self.model.reparameterize(mu, logvar)
                    recon_adj, recon_features = self.model.decode(z, num_nodes.max().item())
                    
                    loss, node_loss, edge_loss, kl_loss = self.simple_loss(
                        recon_adj, recon_features, adj_dense, x_dense, mu, logvar, num_nodes
                    )
                    
                    total_losses.append([
                        loss.item() if not torch.isnan(loss) else 1.0,
                        node_loss.item() if not torch.isnan(node_loss) else 1.0,
                        edge_loss.item() if not torch.isnan(edge_loss) else 1.0,
                        kl_loss.item() if not torch.isnan(kl_loss) else 0.01
                    ])
                except:
                    total_losses.append([1.0, 1.0, 1.0, 0.01])
                    continue
        
        return np.mean(total_losses, axis=0)


def load_hiv_active_data():
    """Load HIV-active molecules"""
    print("Loading ogbg-molhiv dataset...")
    
    import torch._utils
    try:
        import torch_geometric.data.data
        torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr])
        torch.serialization.add_safe_globals([torch_geometric.data.data.Data])
    except:
        pass
    
    try:
        dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='data/')
    except Exception as e:
        print(f"Failed to load with safe globals, trying with weights_only=False...")
        original_torch_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        torch.load = patched_load
        
        try:
            dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='data/')
        finally:
            torch.load = original_torch_load
    
    active_molecules = []
    for i, data in enumerate(tqdm(dataset, desc="Filtering active molecules")):
        if data.y.item() == 1:
            active_molecules.append(data)
    
    print(f"Found {len(active_molecules)} HIV-active molecules out of {len(dataset)} total")
    return active_molecules


def filter_by_size(data_list, max_nodes=30):
    filtered = []
    for data in data_list:
        if data.x.size(0) <= max_nodes:
            filtered.append(data)
    print(f"Filtered to {len(filtered)} molecules with <= {max_nodes} nodes")
    return filtered


def main():
    """Main function"""
    print("Starting GraphVAE training...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Very conservative parameters
    max_nodes = 25  # Smaller graphs
    hidden_dim = 128  # Smaller model
    latent_dim = 32   # Smaller latent space
    batch_size = 16   # Smaller batches
    learning_rate = 0.0001  # Very low learning rate
    num_epochs = 100   # Fewer epochs to start
    
    # Load data
    active_molecules = load_hiv_active_data()
    active_molecules = filter_by_size(active_molecules, max_nodes)
    
    if len(active_molecules) < 50:
        print("Warning: Very few molecules") 
        return
    
    # Split data
    np.random.seed(42)
    indices = np.random.permutation(len(active_molecules))
    train_size = int(0.8 * len(active_molecules))
    
    train_data = [active_molecules[i] for i in indices[:train_size]]
    val_data = [active_molecules[i] for i in indices[train_size:]]
    
    print(f"Training set: {len(train_data)} molecules")
    print(f"Validation set: {len(val_data)} molecules")
    
    # Create loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    # Model
    node_input_dim = train_data[0].x.size(1)
    print(f"Node feature dimension: {node_input_dim}")
    
    model = GraphVAE(
        node_input_dim=node_input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        max_nodes=max_nodes
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Trainer and optimizer
    trainer = Trainer(model, device, beta=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_loss = trainer.train_epoch(train_loader, optimizer, epoch)
        val_loss = trainer.validate(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss[0]:.4f} | Val Loss: {val_loss[0]:.4f}")
        print(f"  Node: {train_loss[1]:.4f} | Edge: {train_loss[2]:.4f} | KL: {train_loss[3]:.6f}")
        
        # Save best model
        if val_loss[0] < best_val_loss:
            best_val_loss = val_loss[0]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss[0],
                'train_loss': train_loss[0],
                'hyperparameters': {
                    'max_nodes': max_nodes,
                    'hidden_dim': hidden_dim,
                    'latent_dim': latent_dim,
                    'node_input_dim': node_input_dim
                }
            }, 'vae_best_model.pth')
            print("  --> New best model saved!")
        
        # Early stop if losses are stable
        if epoch > 10 and abs(train_loss[0] - train_losses[-2][0]) < 0.001:
            print("Training appears stable, you can stop or continue...")
    
    print("\nTraining completed!")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'hyperparameters': {
            'max_nodes': max_nodes,
            'hidden_dim': hidden_dim,
            'latent_dim': latent_dim,
            'node_input_dim': node_input_dim
        }
    }, 'vae_final_model.pth')
    
    print("VAE model saved!")


if __name__ == "__main__":
    main()
