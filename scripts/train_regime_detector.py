
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

from src.learning.market_regime_detection import RegimeDetectorModel, MarketRegime

def generate_synthetic_data(num_samples_per_regime=1000):
    """Generates synthetic data for training the regime detection model."""
    features = []
    labels = []

    for i, regime in enumerate(MarketRegime):
        for _ in range(num_samples_per_regime):
            # Generate features with characteristics specific to the regime
            if regime == MarketRegime.VOLATILE:
                feature = np.random.normal(loc=0.5, scale=0.2, size=21)
                feature[0] = np.random.uniform(0.5, 1.0) # High price volatility
            elif regime == MarketRegime.TRENDING:
                feature = np.random.normal(loc=0.3, scale=0.1, size=21)
                feature[1] = np.random.uniform(0.5, 1.0) # High trend strength
            elif regime == MarketRegime.CRASH:
                feature = np.random.normal(loc=0.8, scale=0.2, size=21)
                feature[0] = np.random.uniform(0.7, 1.0) # High price volatility
                feature[2] = np.random.uniform(-1.0, -0.5) # Negative momentum
            else: # NORMAL, RANGE_BOUND, BULL_RUN, RECOVERY, LOW_LIQUIDITY, HIGH_IMPACT_NEWS
                feature = np.random.normal(loc=0.2, scale=0.1, size=21)
            
            features.append(feature)
            labels.append(i)

    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

def train_model(model, dataloader, num_epochs=10):
    """Trains the regime detection model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs['regime_logits'], labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def main():
    """Main function to train and save the model."""
    # Generate data
    features, labels = generate_synthetic_data()
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    model = RegimeDetectorModel(num_classes=len(MarketRegime))

    # Train model
    train_model(model, dataloader)

    # Save model
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(model.state_dict(), "models/regime_detector_model.pth")
    print("Model trained and saved to models/regime_detector_model.pth")

if __name__ == "__main__":
    main()
