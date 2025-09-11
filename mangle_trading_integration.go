package main

import (
	"fmt"
	"log"

	"github.com/google/mangle/engine"
	"github.com/google/mangle/factstore"
	"github.com/google/mangle/parse"
)

func main() {
	fmt.Println("🔥 Mangle Deductive Database Integration for Trading Bot")
	fmt.Println("======================================================")
	fmt.Println()
	fmt.Println("Project: Self Learning, Self Adapting, Self Healing Neural Network")
	fmt.Println("         of a Fully Autonomous Algorithmic Crypto High leveraged")
	fmt.Println("         Futures Scalping and Trading bot")
	fmt.Println()

	// Mangle program as a string (Datalog-style rules)
	tradingRules := `
		% ============================================
		% TRADING BOT KNOWLEDGE BASE - Facts
		% ============================================

		% Market regime classifications
		market_regime("BTC/USDT", "HIGHLY_VOLATILE").
		market_regime("ETH/USDT", "MODERATE_VOLATILE").
		market_regime("BNB/USDT", "NORMAL").
		market_regime("SOL/USDT", "BULLISH").
		market_regime("ADA/USDT", "BEARISH").

		% Risk assessment scores (0.0 to 1.0)
		risk_score("BTC/USDT", 0.92).
		risk_score("ETH/USDT", 0.78).
		risk_score("BNB/USDT", 0.65).
		risk_score("SOL/USDT", 0.88).
		risk_score("ADA/USDT", 0.71).

		% Asset correlations (coefficient between -1.0 and 1.0)
		correlation("BTC/USDT", "ETH/USDT", 0.87).
		correlation("ETH/USDT", "BNB/USDT", 0.73).
		correlation("BTC/USDT", "BNB/USDT", 0.69).
		correlation("BTC/USDT", "SOL/USDT", 0.81).
		correlation("ETH/USDT", "SOL/USDT", 0.76).

		% Strategy performance metrics (win rate)
		strategy_performance("scalping", 0.74).
		strategy_performance("momentum", 0.68).
		strategy_performance("mean_reversion", 0.61).
		strategy_performance("arbitrage", 0.69).
		strategy_performance("market_making", 0.58).

		% Leverage limits by market regime
		max_leverage("HIGHLY_VOLATILE", 2.0).
		max_leverage("MODERATE_VOLATILE", 3.5).
		max_leverage("NORMAL", 5.0).
		max_leverage("BULLISH", 8.0).
		max_leverage("BEARISH", 2.5).

		% AI model performance (accuracy scores)
		model_accuracy("neural_network", 0.81).
		model_accuracy("lstm", 0.76).
		model_accuracy("transformer", 0.84).
		model_accuracy("ensemble", 0.89).
		model_accuracy("reinforcement_learning", 0.79).

		% Current portfolio positions (example data)
		position("BTC/USDT", 0.4).
		position("ETH/USDT", 0.3).
		position("BNB/USDT", 0.2).
		position("SOL/USDT", 0.1).

		% ============================================
		% DEDUCTIVE REASONING RULES
		% ============================================

		% Rule 1: Detect high correlation risk between assets
		high_correlation_risk(Asset1, Asset2) :-
			correlation(Asset1, Asset2, Corr),
			Corr > 0.8.

		% Rule 2: Calculate risk-adjusted leverage
		recommended_leverage(Asset, Leverage) :-
			market_regime(Asset, Regime),
			max_leverage(Regime, MaxLev),
			risk_score(Asset, Risk),
			Leverage = MaxLev * (1.0 - Risk).

		% Rule 3: Identify assets requiring hedging
		needs_hedging(Asset) :-
			risk_score(Asset, Risk),
			Risk > 0.8.

		% Rule 4: Determine optimal strategy by market regime
		optimal_strategy(Asset, "scalping") :-
			market_regime(Asset, "HIGHLY_VOLATILE"),
			strategy_performance("scalping", Perf),
			Perf > 0.7.

		optimal_strategy(Asset, "momentum") :-
			market_regime(Asset, "BULLISH"),
			strategy_performance("momentum", Perf),
			Perf > 0.65.

		optimal_strategy(Asset, "mean_reversion") :-
			market_regime(Asset, "BEARISH"),
			strategy_performance("mean_reversion", Perf),
			Perf > 0.6.

		% Rule 5: Portfolio risk assessment
		portfolio_risk_alert() :-
			high_correlation_risk(A1, A2),
			position(A1, P1),
			position(A2, P2),
			P1 > 0.1,
			P2 > 0.1.

		% Rule 6: AI model selection
		best_model(Model, Accuracy) :-
			model_accuracy(Model, Accuracy).

		% Rule 7: Position sizing recommendations
		position_limit(Asset, Limit) :-
			risk_score(Asset, Risk),
			Limit = 1.0 - Risk.

		% Rule 8: Strategy conflict detection
		strategy_conflict(Asset, Strategy1, Strategy2) :-
			optimal_strategy(Asset, Strategy1),
			optimal_strategy(Asset, Strategy2),
			Strategy1 != Strategy2.

		% ============================================
		% AGGREGATION QUERIES
		% ============================================

		% Count high-risk assets
		high_risk_asset_count(Count) :-
			risk_score(Asset, Risk) |> do fn:filter(Risk > 0.7), let Count = fn:Count().

		% Average risk by regime
		avg_risk_by_regime(Regime, AvgRisk) :-
			market_regime(Asset, Regime),
			risk_score(Asset, Risk) |> do fn:group_by(), let AvgRisk = fn:Average().

		% Best performing strategy
		best_overall_strategy(Strategy, Performance) :-
			strategy_performance(Strategy, Performance) |> do fn:max(Performance).

		% Portfolio diversification score
		diversification_score(Score) :-
			correlation(A1, A2, Corr) |> do fn:filter(Corr < 0.6), let LowCorrCount = fn:Count(),
			correlation(A1, A2, Corr) |> do fn:filter(Corr > 0.8), let HighCorrCount = fn:Count(),
			Score = LowCorrCount / (LowCorrCount + HighCorrCount + 0.001).
	`

	fmt.Println("📚 Initializing Mangle Deductive Database...")
	fmt.Println("===========================================")

	// Parse the Mangle program
	prog, err := parse.ParseString("trading_bot.mgl", tradingRules)
	if err != nil {
		log.Fatalf("Failed to parse Mangle program: %v", err)
	}

	// Create in-memory fact store
	store := factstore.NewInMemoryStore()

	// Create Mangle engine
	eng := engine.NewEngine(store)

	// Evaluate the program (load facts and rules)
	if err := eng.Eval(prog); err != nil {
		log.Fatalf("Failed to evaluate Mangle program: %v", err)
	}

	fmt.Println("✅ Mangle program loaded successfully!")
	fmt.Println()

	// ============================================
	// QUERY EXECUTION - Trading Bot Intelligence
	// ============================================

	fmt.Println("🔍 Executing Trading Bot Intelligence Queries:")
	fmt.Println("=============================================")

	queries := []struct {
		name        string
		query       string
		description string
	}{
		{
			"High Correlation Risk Analysis",
			`high_correlation_risk(A1, A2)`,
			"Identify asset pairs with dangerous correlation levels",
		},
		{
			"Leverage Recommendations",
			`recommended_leverage(Asset, Leverage)`,
			"Calculate risk-adjusted leverage for each asset",
		},
		{
			"Hedging Requirements",
			`needs_hedging(Asset)`,
			"Find assets that require hedging due to high risk",
		},
		{
			"Optimal Strategies",
			`optimal_strategy(Asset, Strategy)`,
			"Determine best trading strategies by market regime",
		},
		{
			"Portfolio Risk Assessment",
			`portfolio_risk_alert()`,
			"Check if current portfolio has risk concentration",
		},
		{
			"AI Model Selection",
			`best_model(Model, Accuracy)`,
			"Find the best performing AI model",
		},
		{
			"Position Limits",
			`position_limit(Asset, Limit)`,
			"Calculate safe position limits based on risk",
		},
		{
			"Strategy Conflicts",
			`strategy_conflict(Asset, S1, S2)`,
			"Detect conflicting trading strategies",
		},
	}

	for i, q := range queries {
		fmt.Printf("\n%d️⃣  %s:\n", i+1, q.name)
		fmt.Printf("   📝 %s\n", q.description)

		results, err := eng.Query(parse.MustParseDecl(q.query))
		if err != nil {
			fmt.Printf("   ❌ Query error: %v\n", err)
			continue
		}

		if len(results) == 0 {
			fmt.Printf("   📊 No results found\n")
		} else {
			fmt.Printf("   📊 Found %d results:\n", len(results))
			for j, result := range results {
				if j < 5 { // Limit output to first 5 results
					fmt.Printf("      • %v\n", result)
				}
			}
			if len(results) > 5 {
				fmt.Printf("      ... and %d more results\n", len(results)-5)
			}
		}
	}

	// ============================================
	// AGGREGATION ANALYSIS
	// ============================================

	fmt.Println("\n📈 Aggregation Analysis:")
	fmt.Println("========================")

	aggQueries := []struct {
		name        string
		query       string
		description string
	}{
		{
			"High-Risk Asset Count",
			`high_risk_asset_count(Count)`,
			"Count assets with risk score > 0.7",
		},
		{
			"Average Risk by Regime",
			`avg_risk_by_regime(Regime, AvgRisk)`,
			"Calculate average risk for each market regime",
		},
		{
			"Best Overall Strategy",
			`best_overall_strategy(Strategy, Performance)`,
			"Find the best performing strategy across all conditions",
		},
		{
			"Portfolio Diversification",
			`diversification_score(Score)`,
			"Assess portfolio diversification quality",
		},
	}

	for i, q := range aggQueries {
		fmt.Printf("\n%d️⃣  %s:\n", i+1, q.name)
		fmt.Printf("   📝 %s\n", q.description)

		results, err := eng.Query(parse.MustParseDecl(q.query))
		if err != nil {
			fmt.Printf("   ❌ Query error: %v\n", err)
			continue
		}

		if len(results) == 0 {
			fmt.Printf("   📊 No results found\n")
		} else {
			for _, result := range results {
				fmt.Printf("   📊 Result: %v\n", result)
			}
		}
	}

	// ============================================
	// TRADING BOT RECOMMENDATIONS
	// ============================================

	fmt.Println("\n🎯 Trading Bot Integration Recommendations:")
	fmt.Println("==========================================")

	fmt.Println("\n🔬 Research & Development:")
	fmt.Println("   • Use Mangle rules to validate trading strategies")
	fmt.Println("   • Implement real-time risk assessment using correlation analysis")
	fmt.Println("   • Create dynamic position sizing based on regime detection")
	fmt.Println("   • Develop automated strategy selection using performance metrics")

	fmt.Println("\n⚙️ System Architecture:")
	fmt.Println("   • Embed Mangle engine as the reasoning core")
	fmt.Println("   • Load market data as facts in real-time")
	fmt.Println("   • Execute queries to get trading decisions")
	fmt.Println("   • Use aggregation for portfolio optimization")

	fmt.Println("\n📊 Risk Management:")
	fmt.Println("   • Continuous correlation monitoring")
	fmt.Println("   • Automated hedging signal generation")
	fmt.Println("   • Dynamic leverage adjustment")
	fmt.Println("   • Portfolio risk alerts and mitigation")

	fmt.Println("\n🚀 Performance Benefits:")
	fmt.Println("   • Deterministic decision making")
	fmt.Println("   • Auditable trading logic")
	fmt.Println("   • Fast query execution for real-time decisions")
	fmt.Println("   • Scalable rule-based system")

	fmt.Println("\n💡 Integration Example:")
	fmt.Println("   // In your Go trading bot:")
	fmt.Println("   results, _ := mangleEngine.Query(parse.MustParseDecl(\"recommended_leverage(\\\"BTC/USDT\\\", Leverage)\"))")
	fmt.Println("   leverage := results[0][\"Leverage\"]")
	fmt.Println("   // Use leverage value for position sizing")

	fmt.Println("\n✅ Mangle Integration Complete!")
	fmt.Println("================================")
	fmt.Println("\nMangle provides the logical reasoning foundation for your autonomous")
	fmt.Println("trading bot, enabling data-driven decisions based on market conditions,")
	fmt.Println("risk metrics, and strategy performance.")
}
