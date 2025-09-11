"""
Simple import test for enhanced components
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all enhanced components can be imported"""
    try:
        # Test meta learning imports
        from src.learning.meta_learning.meta_learning_engine import MetaLearningEngine, MetaLearningConfig
        print("✓ MetaLearningEngine imported successfully")
        
        # Test market adaptation imports
        from src.learning.self_adaptation.market_adaptation import AdvancedMarketAdaptation, AdaptationConfig
        print("✓ AdvancedMarketAdaptation imported successfully")
        
        # Test self healing imports
        from src.learning.self_healing.self_healing_engine import EnhancedSelfHealingEngine, HealingConfig
        print("✓ EnhancedSelfHealingEngine imported successfully")
        
        # Test neural network imports
        from src.learning.neural_networks.enhanced_neural_network import EnhancedTradingNeuralNetwork, NetworkConfig
        print("✓ EnhancedTradingNeuralNetwork imported successfully")
        
        # Test HFT engine imports
        from src.trading.hft_engine.ultra_low_latency_engine import UltraLowLatencyTradingEngine, ExecutionConfig
        print("✓ UltraLowLatencyTradingEngine imported successfully")
        
        # Test learning manager imports
        from src.learning.learning_manager import LearningManager, LearningConfig
        print("✓ LearningManager imported successfully")
        
        # Test API router imports
        from src.api.routers.enhanced_trading import enhanced_trading_router
        print("✓ Enhanced trading router imported successfully")
        
        print("\n🎉 All enhanced components imported successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)