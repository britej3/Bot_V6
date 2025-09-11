import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DRLAgent(nn.Module):
    """
    Basic Deep Reinforcement Learning Agent with a simple neural network.
    This is a foundational placeholder for a DRL agent.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64, learning_rate: float = 0.001):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss() # Example loss for Q-learning or similar

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(state)

    def choose_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """
        Chooses an action based on the current state using an epsilon-greedy policy.
        """
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim) # Explore
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                q_values = self.forward(state_tensor)
            return torch.argmax(q_values).item() # Exploit

    def learn(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Performs a learning step (e.g., Q-learning update).
        This is a simplified placeholder.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

        q_values = self.forward(state_tensor)
        next_q_values = self.forward(next_state_tensor)

        target_q_values = q_values.clone()
        target_q_values[action] = reward + (0.99 * torch.max(next_q_values) * (1 - done))

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        logger.info(f"DRL Agent learned. Loss: {loss.item():.4f}")

    def save(self, path: str):
        """Saves the agent's model parameters."""
        torch.save(self.state_dict(), path)
        logger.info(f"DRL Agent model saved to {path}")

    def load(self, path: str):
        """Loads the agent's model parameters."""
        self.load_state_dict(torch.load(path))
        logger.info(f"DRL Agent model loaded from {path}")
