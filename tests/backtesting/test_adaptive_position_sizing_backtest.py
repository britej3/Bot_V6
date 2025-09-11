
import pytest
import pandas as pd
from unittest.mock import Mock
import numpy as np

from src.learning.adaptive_risk_management import AdaptiveRiskManager, MarketCondition, MarketRegime, RiskProfile
from src.learning.position_sizer import PositionSizer

@pytest.fixture
def historical_data():
    """Load historical data for backtesting."""
    return pd.read_csv("tests/fixtures/historical_data.csv", parse_dates=["timestamp"])

@pytest.fixture
def adaptive_risk_manager():
    """Initialize the AdaptiveRiskManager for testing."""
    return AdaptiveRiskManager()

def run_backtest(historical_data, risk_manager):
    """
    A simple backtester to validate adaptive position sizing.
    """
    portfolio_value = 100000  # Initial portfolio value
    position = 0  # Current position in the asset
    pnl = 0
    trade_history = []
    portfolio_history = []

    # Simple moving average crossover strategy
    historical_data['sma_short'] = historical_data['close'].rolling(window=5).mean()
    historical_data['sma_long'] = historical_data['close'].rolling(window=20).mean()

    for i, row in historical_data.iterrows():
        if i < 20:  # Skip initial rows where SMAs are not available
            continue

        # 1. Determine market condition (simplified for this test)
        volatility = historical_data['close'].pct_change().rolling(window=20).std().iloc[i] * np.sqrt(365) # Annualized
        if pd.isna(volatility):
            volatility = 0.2 # Default volatility
            
        market_regime = MarketRegime.NORMAL
        if volatility > 0.4:
            market_regime = MarketRegime.VOLATILE
        
        market_condition = MarketCondition(
            regime=market_regime,
            volatility=volatility,
            trend_strength=0.5, # Dummy value
            correlation_level=0.5, # Dummy value
            liquidity_score=0.8, # Dummy value
            confidence=0.8 # Dummy value
        )
        
        # 2. Generate trading signal
        signal = 0  # 1 for buy, -1 for sell, 0 for hold
        if row['sma_short'] > row['sma_long'] and position <= 0:
            signal = 1
        elif row['sma_short'] < row['sma_long'] and position >= 0:
            signal = -1

        # 3. Calculate position size using AdaptiveRiskManager
        if signal != 0:
            risk_profile = risk_manager.risk_profiles.get(market_regime, risk_manager.risk_profiles[MarketRegime.NORMAL])
            
            position_sizer = PositionSizer()
            
            # For simplicity, we assume a fixed stop-loss of 2% from the entry price
            stop_loss_price = row['close'] * (1 - 0.02) if signal == 1 else row['close'] * (1 + 0.02)

            calculated_size = position_sizer.calculate_position_size(
                portfolio_value=portfolio_value,
                entry_price=row['close'],
                stop_loss_price=stop_loss_price,
                market_condition=market_condition,
                risk_profile=risk_profile
            )
            
            # 4. Simulate trade
            if signal == 1: # Buy
                position = calculated_size
            elif signal == -1: # Sell
                position = -calculated_size
            
            trade_history.append({
                "timestamp": row["timestamp"],
                "signal": signal,
                "price": row["close"],
                "size": position
            })

        # 5. Update portfolio value
        if i + 1 < len(historical_data):
            price_change = historical_data['close'].iloc[i+1] - row['close']
            pnl = position * price_change
            portfolio_value += pnl
            portfolio_history.append(portfolio_value)

    return {
        "final_portfolio_value": portfolio_value,
        "trade_history": trade_history,
        "portfolio_history": portfolio_history
    }


def test_adaptive_position_sizing_backtest(historical_data, adaptive_risk_manager):
    """
    Test the adaptive position sizing through a backtest.
    """
    results = run_backtest(historical_data, adaptive_risk_manager)

    # Calculate performance metrics
    portfolio_history = pd.Series(results["portfolio_history"])
    returns = portfolio_history.pct_change().dropna()
    
    sharpe_ratio = 0
    if returns.std() > 0:
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) # Annualized

    # Calculate max drawdown
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    # Assertions
    assert results["final_portfolio_value"] > 100000, "The strategy should be profitable."
    assert sharpe_ratio > 0.5, f"Sharpe ratio should be positive and greater than 0.5, but it is {sharpe_ratio}"
    assert max_drawdown > -0.2, f"Max drawdown should be less than 20%, but it is {max_drawdown * 100}%"
    assert len(results["trade_history"]) > 0, "The backtest should have generated trades."
