
import pytest
import asyncio
from unittest.mock import patch, AsyncMock

from src.learning.adaptive_risk_integration_service import AdaptiveRiskIntegrationService

@pytest.fixture
async def integration_service():
    """Create a test integration service."""
    service = AdaptiveRiskIntegrationService()
    await service.initialize()
    yield service
    await service.stop()

@pytest.mark.asyncio
async def test_high_volume_of_trades(integration_service):
    """Test the system's performance under a high volume of trades."""
    signals = []
    for i in range(1000):
        signals.append({
            'symbol': 'BTC/USDT',
            'action': 'buy' if i % 2 == 0 else 'sell',
            'quantity': 0.01,
            'price': 50000 + i * 0.1,
            'confidence': 0.8
        })

    tasks = [integration_service.process_trade_signal(signal) for signal in signals]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful_results = [r for r in results if isinstance(r, dict) and 'error' not in r]
    assert len(successful_results) > 900, "The system should handle a high volume of trades with minimal errors."

@pytest.mark.asyncio
async def test_extreme_market_movements(integration_service):
    """Test the system's behavior during extreme market movements."""
    # Simulate a market crash
    crash_signals = []
    for i in range(100):
        crash_signals.append({
            'symbol': 'BTC/USDT',
            'action': 'sell',
            'quantity': 0.1,
            'price': 50000 - i * 100,
            'confidence': 0.9
        })

    tasks = [integration_service.process_trade_signal(signal) for signal in crash_signals]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # The risk management system should reject most of these trades or reduce position sizes
    approved_trades = [r for r in results if isinstance(r, dict) and r.get('risk_approved')] 
    assert len(approved_trades) < 50, "The risk management system should intervene during a market crash."

@pytest.mark.asyncio
@patch('src.monitoring.comprehensive_monitoring.ComprehensiveMonitoringSystem.check_exchange_connectivity', new_callable=AsyncMock)
async def test_network_disconnect(mock_check_exchange_connectivity, integration_service):
    """Test the system's behavior during a network disconnect."""
    mock_check_exchange_connectivity.return_value = False

    # The system should stop trading when the exchange is disconnected
    signal = {
        'symbol': 'BTC/USDT',
        'action': 'buy',
        'quantity': 0.1,
        'price': 50000,
        'confidence': 0.8
    }
    
    await asyncio.sleep(1) # allow monitoring to update

    result = await integration_service.process_trade_signal(signal)
    assert result.get('risk_approved') is False, "The system should not approve trades during a network disconnect."
    assert 'Exchange connectivity issue' in result.get('reason', ''), "The reason for rejection should be clear."

@pytest.mark.asyncio
@patch('src.trading.hft_engine.HighFrequencyTradingEngine.execute_trade', new_callable=AsyncMock)
async def test_component_failure(mock_execute_trade, integration_service):
    """Test the system's behavior when a critical component fails."""
    mock_execute_trade.side_effect = Exception("Trade execution failed")

    signal = {
        'symbol': 'BTC/USDT',
        'action': 'buy',
        'quantity': 0.1,
        'price': 50000,
        'confidence': 0.8
    }

    result = await integration_service.process_trade_signal(signal)

    assert 'error' in result, "The system should report an error when a component fails."
    assert "Trade execution failed" in result['error'], "The error message should be propagated."

    # The system should remain operational
    status = integration_service.get_status()
    assert status['is_running'] is True, "The system should remain operational after a component failure."
