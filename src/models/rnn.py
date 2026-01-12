"""RNN baseline for next-step IMU prediction using PyTorch.

Architecture:
  - Input: full window (128, 6)
  - LSTM: 1 layer, hidden size 32
  - Output: last hidden state → linear head → (6,)
  
This is a minimal, interpretable baseline that should outperform persistence.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseModel


class RNNNet(nn.Module):
    """PyTorch LSTM module for next-step prediction."""
    
    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        
        # LSTM: input_size=6, hidden_size=32, 1 layer
        self.lstm = nn.LSTM(
            input_size=6,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # Linear head: hidden_size -> 6
        self.head = nn.Linear(hidden_size, 6)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch, 127, 6)
        
        Returns:
            shape (batch, 6)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)  # h_n: (1, batch, hidden_size)
        
        # Use last hidden state
        h_last = h_n[-1]  # (batch, hidden_size)
        
        # Linear output
        out = self.head(h_last)  # (batch, 6)
        
        return out


class RNNBaseline(BaseModel):
    """LSTM baseline for next-step IMU prediction.
    
    Feeds full window, uses last hidden state + linear head.
    Trains on MSE loss for a fixed number of epochs.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name="rnn_baseline")
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        
        self.model: Optional[RNNNet] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the RNN on the window→next-step task.
        
        Args:
            X: shape (N, T, 6) - windows of variable length T
            y: shape (N, 6) - next-step targets
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        if X.ndim != 3 or X.shape[2] != 6 or y.shape != (X.shape[0], 6):
            raise ValueError(
                f"Expected X shape (N, T, 6) and y shape (N, 6), "
                f"got {X.shape} and {y.shape}"
            )
        
        # Ensure we're working with copies to avoid accidental modification
        X = np.array(X, dtype=np.float32, copy=True)
        y = np.array(y, dtype=np.float32, copy=True)
        
        # Convert to torch
        X_torch = torch.from_numpy(X).float()
        y_torch = torch.from_numpy(y).float()
        
        dataset = TensorDataset(X_torch, y_torch)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        self.model = RNNNet(hidden_size=self.hidden_size).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                preds = self.model(batch_X)
                loss = loss_fn(preds, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(batch_X)
            
            avg_loss = epoch_loss / len(dataset)
            if (epoch + 1) % max(1, self.epochs // 5) == 0 or epoch == 0:
                print(f"  Epoch {epoch + 1}/{self.epochs} - Loss: {avg_loss:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict next-step IMU readings.
        
        Args:
            X: shape (N, T, 6) - windows of variable length T
        
        Returns:
            shape (N, 6)
        """
        if self.model is None:
            raise RuntimeError("Model not fitted yet")
        
        if X.ndim != 3 or X.shape[2] != 6:
            raise ValueError(f"Expected shape (N, T, 6), got {X.shape}")
        
        # Ensure we're working with a copy
        X = np.array(X, dtype=np.float32, copy=True)
        X_torch = torch.from_numpy(X).float().to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_torch)  # (N, 6)
        
        return preds.cpu().numpy().astype(np.float32)
