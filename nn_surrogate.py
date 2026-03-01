"""
Neural Network Surrogate for Monte Carlo Pricing

This module implements neural network approximations for option pricing:
- Train NN to approximate MC simulation results
- Achieve 1000x+ speedup for inference
- Support multiple option types and models

References:
    - Horvath et al. (2021). Deep Learning for Option Pricing.
    - Ferguson & Green (2018). Deeply Learning Derivatives.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    optim = None
    TORCH_AVAILABLE = False


@dataclass
class SurrogateConfig:
    """Configuration for neural network surrogate."""
    hidden_layers: List[int] = None
    activation: str = "relu"
    dropout: float = 0.1
    batch_size: int = 256
    epochs: int = 100
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    device: str = "auto"

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [128, 128, 64, 32]
        if self.device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"


@dataclass
class TrainingResult:
    """Result of surrogate training."""
    train_loss: float
    val_loss: float
    epochs_trained: int
    best_epoch: int
    train_time: float
    mean_abs_error: float
    mean_rel_error: float


class OptionPricingSurrogate:
    """Neural network surrogate for option pricing."""

    def __init__(self, config: Optional[SurrogateConfig] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for neural network surrogate")

        self.config = config or SurrogateConfig()
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.is_trained = False

    def _build_model(self, input_dim: int, output_dim: int = 1) -> nn.Module:
        """Build neural network architecture."""
        layers = []
        prev_dim = input_dim

        for hidden_dim in self.config.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if self.config.activation == "relu":
                layers.append(nn.ReLU())
            elif self.config.activation == "tanh":
                layers.append(nn.Tanh())
            elif self.config.activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.1))
            elif self.config.activation == "silu":
                layers.append(nn.SiLU())

            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize input features."""
        if fit or self.scaler_X is None:
            self.scaler_X = {
                'mean': X.mean(axis=0),
                'std': X.std(axis=0) + 1e-8
            }
        return (X - self.scaler_X['mean']) / self.scaler_X['std']

    def _normalize_y(self, y: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize target values."""
        if fit or self.scaler_y is None:
            self.scaler_y = {
                'mean': y.mean(),
                'std': y.std() + 1e-8
            }
        return (y - self.scaler_y['mean']) / self.scaler_y['std']

    def _denormalize_y(self, y: np.ndarray) -> np.ndarray:
        """Denormalize target values."""
        return y * self.scaler_y['std'] + self.scaler_y['mean']

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True
    ) -> TrainingResult:
        """
        Train the neural network surrogate.

        Args:
            X: Input features (S0, K, r, sigma, T, ...)
            y: Target prices
            verbose: Print training progress

        Returns:
            TrainingResult with training metrics
        """
        import time
        start_time = time.time()

        # Normalize
        X_norm = self._normalize(X, fit=True)
        y_norm = self._normalize_y(y.reshape(-1, 1), fit=True)

        # Train/validation split
        n = len(X)
        n_val = int(n * self.config.validation_split)
        indices = np.random.permutation(n)
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        X_train = torch.FloatTensor(X_norm[train_idx])
        y_train = torch.FloatTensor(y_norm[train_idx])
        X_val = torch.FloatTensor(X_norm[val_idx])
        y_val = torch.FloatTensor(y_norm[val_idx])

        # Move to device
        device = torch.device(self.config.device)
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_val, y_val = X_val.to(device), y_val.to(device)

        # Build model
        self.model = self._build_model(X.shape[1]).to(device)

        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        # Data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)

        # Training loop
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Validate
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = criterion(val_pred, y_val).item()

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

            if patience_counter >= self.config.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        # Restore best model
        self.model.load_state_dict(best_state)
        self.model.to(device)
        self.is_trained = True

        # Compute final metrics
        self.model.eval()
        with torch.no_grad():
            y_pred_norm = self.model(X_val).cpu().numpy()
            y_pred = self._denormalize_y(y_pred_norm)
            y_true = y[val_idx].reshape(-1, 1)

            mae = np.mean(np.abs(y_pred - y_true))
            mre = np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8))

        train_time = time.time() - start_time

        return TrainingResult(
            train_loss=train_loss,
            val_loss=best_val_loss,
            epochs_trained=epoch + 1,
            best_epoch=best_epoch,
            train_time=train_time,
            mean_abs_error=mae,
            mean_rel_error=mre
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict option prices.

        Args:
            X: Input features

        Returns:
            Predicted prices
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        X_norm = self._normalize(X)
        X_tensor = torch.FloatTensor(X_norm).to(self.config.device)

        self.model.eval()
        with torch.no_grad():
            y_norm = self.model(X_tensor).cpu().numpy()

        return self._denormalize_y(y_norm).flatten()

    def save(self, path: str) -> None:
        """Save model to file."""
        torch.save({
            'model_state': self.model.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'config': self.config
        }, path)

    def load(self, path: str, input_dim: int) -> None:
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']
        self.config = checkpoint['config']
        self.model = self._build_model(input_dim).to(self.config.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.is_trained = True


def generate_training_data(
    pricer_func: Callable,
    n_samples: int = 10000,
    param_ranges: Optional[Dict] = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training data by running MC simulations.

    Args:
        pricer_func: Function that takes (S0, K, r, sigma, T) and returns price
        n_samples: Number of samples to generate
        param_ranges: Dict of parameter ranges
        seed: Random seed

    Returns:
        Tuple of (X, y) arrays
    """
    rng = np.random.default_rng(seed)

    if param_ranges is None:
        param_ranges = {
            'S0': (50, 150),
            'K': (50, 150),
            'r': (0.01, 0.10),
            'sigma': (0.10, 0.50),
            'T': (0.1, 2.0)
        }

    X = np.zeros((n_samples, 5))
    y = np.zeros(n_samples)

    for i in range(n_samples):
        X[i, 0] = rng.uniform(*param_ranges['S0'])  # S0
        X[i, 1] = rng.uniform(*param_ranges['K'])   # K
        X[i, 2] = rng.uniform(*param_ranges['r'])   # r
        X[i, 3] = rng.uniform(*param_ranges['sigma'])  # sigma
        X[i, 4] = rng.uniform(*param_ranges['T'])   # T

        y[i] = pricer_func(X[i, 0], X[i, 1], X[i, 2], X[i, 3], X[i, 4])

    return X, y


if __name__ == "__main__":
    print("=" * 60)
    print("Neural Network Surrogate - Demo")
    print("=" * 60)

    if not TORCH_AVAILABLE:
        print("PyTorch not available. Install with: pip install torch")
        exit()

    # Simple Black-Scholes pricer for demo
    from scipy.stats import norm

    def bs_call(S0, K, r, sigma, T):
        d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

    print("\n1. Generating training data...")
    X, y = generate_training_data(bs_call, n_samples=5000, seed=42)
    print(f"   Generated {len(X)} samples")

    print("\n2. Training surrogate model...")
    config = SurrogateConfig(hidden_layers=[64, 64, 32], epochs=50)
    surrogate = OptionPricingSurrogate(config)
    result = surrogate.train(X, y, verbose=True)

    print(f"\n3. Training Results:")
    print(f"   Final val loss: {result.val_loss:.6f}")
    print(f"   Mean abs error: ${result.mean_abs_error:.4f}")
    print(f"   Mean rel error: {result.mean_rel_error:.2%}")
    print(f"   Training time:  {result.train_time:.2f}s")

    print("\n4. Speed comparison:")
    import time
    test_X = X[:100]

    start = time.time()
    mc_prices = np.array([bs_call(*x) for x in test_X])
    mc_time = time.time() - start

    start = time.time()
    nn_prices = surrogate.predict(test_X)
    nn_time = time.time() - start

    print(f"   MC time:  {mc_time*1000:.2f}ms")
    print(f"   NN time:  {nn_time*1000:.2f}ms")
    print(f"   Speedup:  {mc_time/nn_time:.1f}x")
