"""
Autoencoder model for market regime detection.
This module implements a convolutional autoencoder for unsupervised learning of market regimes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for market regime detection.
    """
    
    def __init__(self, input_channels: int, seq_length: int, latent_dim: int):
        """
        Initialize the convolutional autoencoder.
        
        Args:
            input_channels: Number of input features
            seq_length: Length of input sequences
            latent_dim: Dimension of latent space representation
        """
        super(ConvAutoencoder, self).__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        
        # Calculate sizes for fully connected layers
        self.conv_output_size = seq_length // 8  # After 3 pooling layers with kernel size 2
        self.conv_output_channels = 128  # Final number of channels in encoder
        
        # Encoder
        self.encoder = nn.Sequential(
            # First convolutional block
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # seq_length // 2
            
            # Second convolutional block
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # seq_length // 4
            
            # Third convolutional block
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # seq_length // 8
            
            # Dropout for regularization
            nn.Dropout(0.2)
        )
        
        # Bottleneck (fully connected layers)
        self.fc_encoder = nn.Linear(128 * self.conv_output_size, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, 128 * self.conv_output_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            # First transposed convolutional block
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Second transposed convolutional block
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # Third transposed convolutional block
            nn.ConvTranspose1d(32, input_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input data to latent representation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_channels)
            
        Returns:
            torch.Tensor: Latent representation of shape (batch_size, latent_dim)
        """
        # Reshape input: (batch_size, seq_length, input_channels) -> (batch_size, input_channels, seq_length)
        x = x.permute(0, 2, 1)
        
        # Apply encoder
        x = self.encoder(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Apply bottleneck
        latent = self.fc_encoder(x)
        
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstructed input.
        
        Args:
            latent: Latent representation of shape (batch_size, latent_dim)
            
        Returns:
            torch.Tensor: Reconstructed input of shape (batch_size, seq_length, input_channels)
        """
        # Apply bottleneck
        x = self.fc_decoder(latent)
        
        # Reshape to (batch_size, channels, conv_output_size)
        x = x.view(x.size(0), self.conv_output_channels, self.conv_output_size)
        
        # Apply decoder
        x = self.decoder(x)
        
        # Reshape output: (batch_size, input_channels, seq_length) -> (batch_size, seq_length, input_channels)
        x = x.permute(0, 2, 1)
        
        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_channels)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (reconstructed_input, latent_representation)
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent


class RegimeDetectionLoss(nn.Module):
    """
    Custom loss function for regime detection, combining reconstruction and clustering objectives.
    """
    
    def __init__(self, lambda1: float = 0.1, lambda2: float = 0.05):
        """
        Initialize the regime detection loss.
        
        Args:
            lambda1: Weight for clustering regularization
            lambda2: Weight for temporal consistency regularization
        """
        super(RegimeDetectionLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.mse_loss = nn.MSELoss(reduction='mean')
    
    def forward(self, x: torch.Tensor, reconstructed: torch.Tensor, 
               latent: torch.Tensor, prev_latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the loss.
        
        Args:
            x: Original input tensor
            reconstructed: Reconstructed input tensor
            latent: Latent representation tensor
            prev_latent: Previous batch latent representation for temporal consistency
            
        Returns:
            torch.Tensor: Computed loss value
        """
        # Reconstruction loss
        reconstruction_loss = self.mse_loss(reconstructed, x)
        
        # Clustering regularization (encourage separation in latent space)
        # Compute pairwise distances between latent vectors
        batch_size = latent.size(0)
        if batch_size > 1:
            latent_norm = F.normalize(latent, p=2, dim=1)
            similarity_matrix = torch.mm(latent_norm, latent_norm.t())
            
            # Create target matrix (identity matrix)
            target_matrix = torch.eye(batch_size, device=latent.device)
            
            # Compute clustering loss (encourage orthogonality)
            clustering_loss = F.mse_loss(similarity_matrix, target_matrix)
        else:
            clustering_loss = torch.tensor(0.0, device=latent.device)
        
        # Temporal consistency regularization
        if prev_latent is not None:
            # Encourage smooth transitions in latent space
            temporal_loss = F.mse_loss(latent, prev_latent)
        else:
            temporal_loss = torch.tensor(0.0, device=latent.device)
        
        # Combine losses
        total_loss = reconstruction_loss + self.lambda1 * clustering_loss + self.lambda2 * temporal_loss
        
        return total_loss


def train_autoencoder(model: ConvAutoencoder, train_loader: DataLoader, val_loader: DataLoader,
                     num_epochs: int = 60, learning_rate: float = 0.001,
                     lambda1: float = 0.1, lambda2: float = 0.05,
                     device: torch.device = None) -> ConvAutoencoder:
    """
    Train the autoencoder model.
    
    Args:
        model: ConvAutoencoder model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        lambda1: Weight for clustering regularization
        lambda2: Weight for temporal consistency regularization
        device: Device to train on (CPU or GPU)
        
    Returns:
        ConvAutoencoder: Trained model
    """
    logger.info("Starting autoencoder training")
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Initialize scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Initialize loss function
    criterion = RegimeDetectionLoss(lambda1=lambda1, lambda2=lambda2)
    
    # Initialize early stopping
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        prev_latent = None
        
        for batch_idx, (data,) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            data = data.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed, latent = model(data)
            
            # Compute loss
            loss = criterion(data, reconstructed, latent, prev_latent)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            
            # Update previous latent for temporal consistency
            prev_latent = latent.detach()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        prev_latent = None
        
        with torch.no_grad():
            for batch_idx, (data,) in enumerate(val_loader):
                data = data.to(device)
                
                # Forward pass
                reconstructed, latent = model(data)
                
                # Compute loss
                loss = criterion(data, reconstructed, latent, prev_latent)
                
                # Update statistics
                val_loss += loss.item()
                
                # Update previous latent for temporal consistency
                prev_latent = latent
        
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "/home/ubuntu/adaptive_market_making_implementation/models/autoencoder_best.pth")
            logger.info(f"Saved best model with validation loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    model.load_state_dict(torch.load("/home/ubuntu/adaptive_market_making_implementation/models/autoencoder_best.pth"))
    
    logger.info("Autoencoder training completed")
    
    return model


class RegimeClassifier:
    """
    Classifier for market regimes using autoencoder latent space and Gaussian Mixture Model.
    """
    
    def __init__(self, autoencoder: ConvAutoencoder, gmm: GaussianMixture, 
                device: torch.device = None, smoothing_window: int = 5):
        """
        Initialize the regime classifier.
        
        Args:
            autoencoder: Trained autoencoder model
            gmm: Trained Gaussian Mixture Model
            device: Device for inference
            smoothing_window: Window size for regime label smoothing
        """
        self.autoencoder = autoencoder
        self.gmm = gmm
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.smoothing_window = smoothing_window
        self.autoencoder.to(self.device)
        self.autoencoder.eval()
        self.recent_labels = []
        self.logger = logging.getLogger(__name__ + '.RegimeClassifier')
        
        self.logger.info("Initialized regime classifier")
    
    def encode_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Encode a sequence using the autoencoder.
        
        Args:
            sequence: Input sequence of shape (seq_length, input_channels) or (batch_size, seq_length, input_channels)
            
        Returns:
            np.ndarray: Latent representation
        """
        # Add batch dimension if needed
        if len(sequence.shape) == 2:
            sequence = np.expand_dims(sequence, 0)
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence).to(self.device)
        
        # Encode
        with torch.no_grad():
            latent = self.autoencoder.encode(sequence_tensor)
        
        # Convert back to numpy
        latent_np = latent.cpu().numpy()
        
        return latent_np
    
    def classify(self, sequence: np.ndarray) -> Tuple[int, float]:
        """
        Classify a sequence into a market regime.
        
        Args:
            sequence: Input sequence of shape (seq_length, input_channels) or (batch_size, seq_length, input_channels)
            
        Returns:
            Tuple[int, float]: (regime_label, probability)
        """
        # Encode sequence
        latent = self.encode_sequence(sequence)
        
        # Predict regime using GMM
        labels = self.gmm.predict(latent)
        probs = self.gmm.predict_proba(latent)
        
        # Get label and probability for the first (or only) sequence
        label = labels[0]
        prob = probs[0, label]
        
        # Apply smoothing
        self.recent_labels.append(label)
        if len(self.recent_labels) > self.smoothing_window:
            self.recent_labels.pop(0)
        
        # Majority vote for smoothed label
        from collections import Counter
        smoothed_label = Counter(self.recent_labels).most_common(1)[0][0]
        
        return smoothed_label, prob
    
    def get_regime_distribution(self, sequence: np.ndarray) -> np.ndarray:
        """
        Get the probability distribution over regimes.
        
        Args:
            sequence: Input sequence
            
        Returns:
            np.ndarray: Probability distribution over regimes
        """
        # Encode sequence
        latent = self.encode_sequence(sequence)
        
        # Get probability distribution
        probs = self.gmm.predict_proba(latent)
        
        return probs[0]


def find_optimal_clusters(latent_vectors: np.ndarray, n_components_range: List[int]) -> Tuple[int, GaussianMixture]:
    """
    Find the optimal number of clusters (regimes) using BIC and silhouette score.
    
    Args:
        latent_vectors: Encoded latent vectors
        n_components_range: Range of number of components to try
        
    Returns:
        Tuple[int, GaussianMixture]: (optimal_n_components, trained_gmm)
    """
    logger.info(f"Finding optimal number of clusters in range {n_components_range}")
    
    bic_scores = []
    silhouette_scores = []
    gmm_models = []
    
    for n_components in n_components_range:
        logger.info(f"Trying {n_components} components")
        
        # Train GMM
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            max_iter=100,
            random_state=42
        )
        gmm.fit(latent_vectors)
        
        # Calculate BIC
        bic = gmm.bic(latent_vectors)
        bic_scores.append(bic)
        
        # Calculate silhouette score if n_components > 1
        if n_components > 1:
            labels = gmm.predict(latent_vectors)
            silhouette = silhouette_score(latent_vectors, labels)
            silhouette_scores.append(silhouette)
        else:
            silhouette_scores.append(-1)  # Invalid for n_components=1
        
        gmm_models.append(gmm)
        
        logger.info(f"n_components={n_components}, BIC={bic:.2f}, Silhouette={silhouette_scores[-1]:.4f}")
    
    # Find optimal n_components
    # Normalize scores to [0, 1] range for comparison
    bic_norm = (bic_scores - np.min(bic_scores)) / (np.max(bic_scores) - np.min(bic_scores))
    bic_norm = 1 - bic_norm  # Lower BIC is better, so invert
    
    # Filter out invalid silhouette scores
    valid_indices = [i for i, s in enumerate(silhouette_scores) if s >= 0]
    valid_silhouette = [silhouette_scores[i] for i in valid_indices]
    
    if valid_silhouette:
        # Normalize valid silhouette scores
        silhouette_norm = np.zeros_like(silhouette_scores, dtype=float)
        silhouette_min = min(valid_silhouette)
        silhouette_max = max(valid_silhouette)
        silhouette_range = silhouette_max - silhouette_min
        
        for i in valid_indices:
            if silhouette_range > 0:
                silhouette_norm[i] = (silhouette_scores[i] - silhouette_min) / silhouette_range
            else:
                silhouette_norm[i] = 1.0
        
        # Combine scores (weighted average)
        combined_scores = 0.7 * bic_norm + 0.3 * silhouette_norm
    else:
        # Use only BIC if no valid silhouette scores
        combined_scores = bic_norm
    
    # Find optimal index
    optimal_idx = np.argmax(combined_scores)
    optimal_n_components = n_components_range[optimal_idx]
    optimal_gmm = gmm_models[optimal_idx]
    
    logger.info(f"Optimal number of components: {optimal_n_components}")
    
    return optimal_n_components, optimal_gmm


def visualize_regimes(latent_vectors: np.ndarray, labels: np.ndarray, save_path: str = None) -> None:
    """
    Visualize the identified market regimes in 2D space.
    
    Args:
        latent_vectors: Encoded latent vectors
        labels: Regime labels
        save_path: Path to save the visualization
    """
    from sklearn.decomposition import PCA
    
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Get unique labels
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            latent_2d[mask, 0],
            latent_2d[mask, 1],
            c=[colors[i]],
            label=f"Regime {label}",
            alpha=0.7
        )
    
    plt.title("Market Regimes in Latent Space")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")
    else:
        plt.show()


# Example usage
if __name__ == "__main__":
    import yaml
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    
    # Load configuration
    with open("/home/ubuntu/adaptive_market_making_implementation/config/model_params.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Example: Create and train autoencoder
    # This would use actual data in production
    
    # Simulated data for demonstration
    seq_length = 100
    input_channels = 25
    batch_size = 64
    
    # Create random data
    X_train = np.random.randn(1000, seq_length, input_channels)
    X_val = np.random.randn(200, seq_length, input_channels)
    
    # Convert to PyTorch tensors
    train_tensor = torch.FloatTensor(X_train)
    val_tensor = torch.FloatTensor(X_val)
    
    # Create DataLoaders
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = ConvAutoencoder(
        input_channels=input_channels,
        seq_length=seq_length,
        latent_dim=config['autoencoder']['latent_dim']
    )
    
    # Train model
    trained_model = train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['autoencoder']['num_epochs'],
        learning_rate=config['autoencoder']['learning_rate'],
        lambda1=config['autoencoder']['lambda1'],
        lambda2=config['autoencoder']['lambda2']
    )
    
    # Extract latent vectors
    latent_vectors = []
    with torch.no_grad():
        for batch_idx, (data,) in enumerate(val_loader):
            latent = trained_model.encode(data)
            latent_vectors.append(latent.cpu().numpy())
    
    latent_vectors = np.vstack(latent_vectors)
    
    # Find optimal number of clusters
    n_components, gmm = find_optimal_clusters(
        latent_vectors=latent_vectors,
        n_components_range=config['gmm']['n_components_range']
    )
    
    # Save models
    torch.save(trained_model.state_dict(), "/home/ubuntu/adaptive_market_making_implementation/models/autoencoder_final.pth")
    import joblib
    joblib.dump(gmm, "/home/ubuntu/adaptive_market_making_implementation/models/gmm_regime_model.pkl")
    
    # Visualize regimes
    labels = gmm.predict(latent_vectors)
    visualize_regimes(
        latent_vectors=latent_vectors,
        labels=labels,
        save_path="/home/ubuntu/adaptive_market_making_implementation/models/regime_visualization.png"
    )
