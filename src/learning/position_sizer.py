"""
Position Sizing Component
========================

Implements intelligent position sizing algorithms for the adaptive risk management system.
Uses Kelly criterion, volatility-based adjustments, and market regime awareness.

Key Features:
- Kelly criterion-based position sizing
- Volatility-adjusted sizing
- Market regime aware adjustments
- Risk profile integration
- Drawdown protection

Author: Risk Management Team
Date: 2025-08-25
"""

import logging
import numpy as np
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

try:
    from .adaptive_risk_management import MarketCondition, RiskProfile, MarketRegime
except ImportError:
    # Fallback for direct execution
    MarketRegime = None
    MarketCondition = None
    RiskProfile = None

logger = logging.getLogger(__name__)


class PositionSizingMethod(Enum):
    """Position sizing methods"""
    FIXED = "fixed"
    KELLY = "kelly"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    RISK_PARITY = "risk_parity"


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing"""
    method: PositionSizingMethod = PositionSizingMethod.KELLY
    max_position_size: float = 0.1  # Maximum position size as fraction of portfolio
    min_position_size: float = 0.01  # Minimum position size
    kelly_fraction: float = 0.25    # Fraction of Kelly to use (for safety)
    base_volatility: float = 0.20   # Base volatility for normalization
    lookback_period: int = 252      # Lookback period for calculations


class PositionSizer:
    """
    Intelligent position sizing system that adapts to market conditions
    and integrates with adaptive risk management.
    """
    
    def __init__(self, config: Optional[PositionSizingConfig] = None):
        self.config = config or PositionSizingConfig()
        
        # Historical tracking for Kelly calculation
        self.trade_history = []
        self.win_rate = 0.5  # Default win rate
        self.avg_win_loss_ratio = 1.0  # Default win/loss ratio
        
        logger.info(f"PositionSizer initialized with method: {self.config.method.value}")
    
    def calculate_position_size(self, 
                              portfolio_value: float,
                              entry_price: float,
                              stop_loss_price: float,
                              market_condition: Any,
                              risk_profile: Any) -> float:
        """
        Calculate optimal position size based on various factors.
        
        Args:
            portfolio_value: Current portfolio value
            entry_price: Intended entry price
            stop_loss_price: Stop loss price
            market_condition: Current market condition
            risk_profile: Risk profile for current regime
            
        Returns:
            Position size (number of shares/units)
        """
        
        # Calculate base risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share == 0:
            logger.warning("Risk per share is zero, using minimum position size")
            return self._calculate_min_position(portfolio_value, entry_price)
        
        # Get position sizing method
        if self.config.method == PositionSizingMethod.FIXED:
            size = self._calculate_fixed_size(portfolio_value, entry_price, risk_profile)
        elif self.config.method == PositionSizingMethod.KELLY:
            size = self._calculate_kelly_size(
                portfolio_value, entry_price, risk_per_share, market_condition, risk_profile
            )
        elif self.config.method == PositionSizingMethod.VOLATILITY_ADJUSTED:
            size = self._calculate_volatility_adjusted_size(
                portfolio_value, entry_price, risk_per_share, market_condition, risk_profile
            )
        elif self.config.method == PositionSizingMethod.RISK_PARITY:
            size = self._calculate_risk_parity_size(
                portfolio_value, entry_price, risk_per_share, market_condition, risk_profile
            )
        else:
            size = self._calculate_kelly_size(
                portfolio_value, entry_price, risk_per_share, market_condition, risk_profile
            )
        
        # Apply constraints
        size = self._apply_constraints(size, portfolio_value, entry_price)
        
        logger.debug(f"Position size calculated: {size:.2f} units for ${portfolio_value:.2f} portfolio")
        
        return size
    
    def _calculate_fixed_size(self, portfolio_value: float, entry_price: float, 
                             risk_profile: Any) -> float:
        """Calculate fixed percentage position size"""
        
        if risk_profile and hasattr(risk_profile, 'risk_per_trade'):
            risk_fraction = risk_profile.risk_per_trade
        else:
            risk_fraction = 0.02  # Default 2%
        
        position_value = portfolio_value * risk_fraction
        return position_value / entry_price
    
    def _calculate_kelly_size(self, portfolio_value: float, entry_price: float,
                             risk_per_share: float, market_condition: Any,
                             risk_profile: Any) -> float:
        """Calculate Kelly criterion-based position size"""
        
        # Kelly formula: f = (bp - q) / b
        # where f = fraction of capital to bet
        #       b = odds (reward/risk ratio)
        #       p = probability of winning
        #       q = probability of losing (1 - p)
        
        # Estimate win probability from historical data or use default
        win_prob = self.win_rate if self.win_rate > 0 else 0.5
        
        # Calculate reward to risk ratio (simplified)
        # Assume target is 2x the risk for Kelly calculation
        reward_risk_ratio = 2.0
        
        if hasattr(risk_profile, 'stop_loss_multiplier'):
            reward_risk_ratio = 1.0 / risk_profile.stop_loss_multiplier
        
        # Kelly fraction
        kelly_fraction = (reward_risk_ratio * win_prob - (1 - win_prob)) / reward_risk_ratio
        
        # Apply safety factor
        kelly_fraction *= self.config.kelly_fraction
        
        # Ensure positive
        kelly_fraction = max(0, kelly_fraction)
        
        # Apply volatility adjustment
        if market_condition and hasattr(market_condition, 'volatility'):
            vol_adjustment = self.config.base_volatility / max(market_condition.volatility, 0.01)
            kelly_fraction *= min(vol_adjustment, 2.0)  # Cap adjustment
        
        # Convert to position size
        risk_capital = portfolio_value * kelly_fraction
        return risk_capital / risk_per_share
    
    def _calculate_volatility_adjusted_size(self, portfolio_value: float, entry_price: float,
                                          risk_per_share: float, market_condition: Any,
                                          risk_profile: Any) -> float:
        """Calculate volatility-adjusted position size"""
        
        base_risk_fraction = 0.02  # Base 2% risk
        
        if risk_profile and hasattr(risk_profile, 'risk_per_trade'):
            base_risk_fraction = risk_profile.risk_per_trade
        
        # Adjust for volatility
        if market_condition and hasattr(market_condition, 'volatility'):
            volatility = market_condition.volatility
            vol_adjustment = self.config.base_volatility / max(volatility, 0.01)
            
            # Scale down in high volatility, scale up in low volatility
            adjusted_risk_fraction = base_risk_fraction * min(vol_adjustment, 2.0)
        else:
            adjusted_risk_fraction = base_risk_fraction
        
        # Apply regime-specific multiplier
        if risk_profile and hasattr(risk_profile, 'position_size_multiplier'):
            adjusted_risk_fraction *= risk_profile.position_size_multiplier
        
        risk_capital = portfolio_value * adjusted_risk_fraction
        return risk_capital / risk_per_share
    
    def _calculate_risk_parity_size(self, portfolio_value: float, entry_price: float,
                                   risk_per_share: float, market_condition: Any,
                                   risk_profile: Any) -> float:
        """Calculate risk parity-based position size"""
        
        # Target equal risk contribution
        target_risk_contribution = 0.01  # 1% portfolio risk per position
        
        if risk_profile and hasattr(risk_profile, 'risk_per_trade'):
            target_risk_contribution = risk_profile.risk_per_trade / 2  # Split risk
        
        risk_capital = portfolio_value * target_risk_contribution
        return risk_capital / risk_per_share
    
    def _calculate_min_position(self, portfolio_value: float, entry_price: float) -> float:
        """Calculate minimum viable position size"""
        min_value = portfolio_value * self.config.min_position_size
        return min_value / entry_price
    
    def _apply_constraints(self, size: float, portfolio_value: float, 
                          entry_price: float) -> float:
        """Apply position size constraints"""
        
        # Maximum position size constraint
        max_size = (portfolio_value * self.config.max_position_size) / entry_price
        size = min(size, max_size)
        
        # Minimum position size constraint  
        min_size = (portfolio_value * self.config.min_position_size) / entry_price
        size = max(size, min_size)
        
        # Ensure positive
        size = max(0, size)
        
        return size
    
    def update_trade_result(self, entry_price: float, exit_price: float, 
                           position_size: float, success: bool) -> None:
        """Update historical performance for Kelly calculation"""
        
        pnl = (exit_price - entry_price) * position_size
        
        trade_record = {
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'pnl': pnl,
            'success': success
        }
        
        self.trade_history.append(trade_record)
        
        # Keep only recent history
        if len(self.trade_history) > self.config.lookback_period:
            self.trade_history = self.trade_history[-self.config.lookback_period:]
        
        # Update win rate and win/loss ratio
        self._update_statistics()
        
        logger.debug(f"Trade result updated: PnL={pnl:.2f}, Success={success}")
    
    def _update_statistics(self) -> None:
        """Update win rate and average win/loss ratio from trade history"""
        
        if not self.trade_history:
            return
        
        wins = [t for t in self.trade_history if t['success']]
        losses = [t for t in self.trade_history if not t['success']]
        
        # Update win rate
        self.win_rate = len(wins) / len(self.trade_history)
        
        # Update win/loss ratio
        if wins and losses:
            avg_win = np.mean([t['pnl'] for t in wins])
            avg_loss = np.mean([abs(t['pnl']) for t in losses])
            
            if avg_loss > 0:
                self.avg_win_loss_ratio = avg_win / avg_loss
            else:
                self.avg_win_loss_ratio = 2.0  # Default
        
        logger.debug(f"Statistics updated: Win rate={self.win_rate:.3f}, "
                    f"Win/Loss ratio={self.avg_win_loss_ratio:.3f}")
    
    def get_sizing_statistics(self) -> Dict[str, Any]:
        """Get current position sizing statistics"""
        
        return {
            'method': self.config.method.value,
            'max_position_size': self.config.max_position_size,
            'min_position_size': self.config.min_position_size,
            'kelly_fraction': self.config.kelly_fraction,
            'trade_count': len(self.trade_history),
            'win_rate': self.win_rate,
            'avg_win_loss_ratio': self.avg_win_loss_ratio
        }


# Factory function
def create_position_sizer(method: PositionSizingMethod = PositionSizingMethod.KELLY) -> PositionSizer:
    """Factory function to create position sizer"""
    config = PositionSizingConfig(method=method)
    return PositionSizer(config)


# Demo and testing
if __name__ == "__main__":
    print("📏 Position Sizer Demo")
    print("=" * 25)
    
    # Create position sizer
    sizer = create_position_sizer(PositionSizingMethod.KELLY)
    
    # Mock market condition and risk profile
    class MockMarketCondition:
        volatility = 0.25
        
    class MockRiskProfile:
        risk_per_trade = 0.02
        position_size_multiplier = 1.0
        stop_loss_multiplier = 0.5
    
    market_condition = MockMarketCondition()
    risk_profile = MockRiskProfile()
    
    # Calculate position size
    portfolio_value = 100000.0
    entry_price = 50000.0
    stop_loss_price = 49000.0  # 2% stop loss
    
    position_size = sizer.calculate_position_size(
        portfolio_value=portfolio_value,
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        market_condition=market_condition,
        risk_profile=risk_profile
    )
    
    print(f"Portfolio Value: ${portfolio_value:,.2f}")
    print(f"Entry Price: ${entry_price:,.2f}")
    print(f"Stop Loss Price: ${stop_loss_price:,.2f}")
    print(f"Calculated Position Size: {position_size:.4f} units")
    
    position_value = position_size * entry_price
    portfolio_percentage = (position_value / portfolio_value) * 100
    
    print(f"Position Value: ${position_value:,.2f}")
    print(f"Portfolio Percentage: {portfolio_percentage:.2f}%")
    
    # Show statistics
    stats = sizer.get_sizing_statistics()
    print(f"\nPosition Sizer Statistics: {stats}")
    
    print("\n📏 Position Sizer - Implementation complete")