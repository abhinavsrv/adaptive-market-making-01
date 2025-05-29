"""
Adaptive Market Maker implementation for the adaptive market making strategy.
This module integrates regime detection, spread calculation, and inventory management.
"""

import logging
import numpy as np
import time
from typing import Dict, List, Optional, Union, Any, Tuple
import torch
import joblib
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdaptiveMarketMaker:
    """
    Adaptive Market Maker that integrates regime detection, spread calculation, and inventory management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the adaptive market maker.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__ + '.AdaptiveMarketMaker')
        
        # Load models
        self.autoencoder = self._load_autoencoder(config["model_paths"]["autoencoder"])
        self.gmm = self._load_gmm(config["model_paths"]["gmm"])
        self.scaler = self._load_scaler(config["model_paths"]["scaler"])
        
        # Initialize components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.regime_classifier = self._initialize_regime_classifier()
        self.spread_calculator = self._initialize_spread_calculator()
        self.inventory_manager = self._initialize_inventory_manager()
        self.informed_estimator = self._initialize_informed_estimator()
        
        # Load regime-specific parameters
        self.regime_params = config["regime_specific_params"]
        self.inventory_limits = self.inventory_manager.calculate_inventory_limits(self.regime_params)
        
        # State variables
        self.current_inventory = 0
        self.market_data_buffer = []  # Buffer for features
        self.order_history = []  # For informed trading estimation
        self.return_history = []  # For informed trading estimation
        self.current_regime = "normal"  # Default regime
        self.current_regime_prob = 1.0
        self.current_informed_prob = 0.2  # Default informed trading probability
        
        # Trading parameters
        self.tick_size = config["trading"]["tick_size"]
        self.contract_multiplier = config["trading"]["contract_multiplier"]
        self.max_position = config["trading"]["max_position"]
        self.target_inventory = config["trading"]["target_inventory"]
        
        # Performance tracking
        self.trades = []
        self.quotes = []
        self.pnl_history = []
        self.regime_history = []
        
        self.logger.info("Initialized adaptive market maker")
    
    def _load_autoencoder(self, path: str):
        """
        Load the autoencoder model.
        
        Args:
            path: Path to the model file
            
        Returns:
            Loaded autoencoder model
        """
        try:
            self.logger.info(f"Loading autoencoder from {path}")
            
            # Import here to avoid circular imports
            from src.regime_detection.autoencoder import ConvAutoencoder
            
            # Get model parameters from config
            input_channels = self.config["model_params"]["autoencoder"]["input_channels"]
            seq_length = self.config["model_params"]["autoencoder"]["seq_length"]
            latent_dim = self.config["model_params"]["autoencoder"]["latent_dim"]
            
            # Initialize model architecture
            model = ConvAutoencoder(input_channels, seq_length, latent_dim)
            
            # Load state dict
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()
            
            self.logger.info("Autoencoder loaded successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load autoencoder: {str(e)}")
            raise
    
    def _load_gmm(self, path: str):
        """
        Load the GMM model.
        
        Args:
            path: Path to the model file
            
        Returns:
            Loaded GMM model
        """
        try:
            self.logger.info(f"Loading GMM from {path}")
            
            # Load using joblib
            gmm = joblib.load(path)
            
            self.logger.info(f"GMM loaded successfully with {gmm.n_components} components")
            return gmm
            
        except Exception as e:
            self.logger.error(f"Failed to load GMM: {str(e)}")
            raise
    
    def _load_scaler(self, path: str):
        """
        Load the feature scaler.
        
        Args:
            path: Path to the scaler file
            
        Returns:
            Loaded scaler
        """
        try:
            self.logger.info(f"Loading scaler from {path}")
            
            # Load using joblib
            scaler = joblib.load(path)
            
            self.logger.info("Scaler loaded successfully")
            return scaler
            
        except Exception as e:
            self.logger.error(f"Failed to load scaler: {str(e)}")
            raise
    
    def _initialize_regime_classifier(self):
        """
        Initialize the regime classifier.
        
        Returns:
            Initialized regime classifier
        """
        try:
            self.logger.info("Initializing regime classifier")
            
            # Import here to avoid circular imports
            from src.regime_detection.autoencoder import RegimeClassifier
            
            # Get parameters from config
            smoothing_window = self.config["model_params"]["regime_classifier"]["smoothing_window"]
            
            # Initialize classifier
            classifier = RegimeClassifier(
                autoencoder=self.autoencoder,
                gmm=self.gmm,
                device=self.device,
                smoothing_window=smoothing_window
            )
            
            self.logger.info("Regime classifier initialized successfully")
            return classifier
            
        except Exception as e:
            self.logger.error(f"Failed to initialize regime classifier: {str(e)}")
            raise
    
    def _initialize_spread_calculator(self):
        """
        Initialize the spread calculator.
        
        Returns:
            Initialized spread calculator
        """
        try:
            self.logger.info("Initializing spread calculator")
            
            # Import here to avoid circular imports
            from src.strategy.spread_calculator import OptimalSpreadCalculator
            
            # Get parameters from config
            risk_aversion = self.config["strategy_params"]["general"]["risk_aversion"]
            
            # Initialize calculator
            calculator = OptimalSpreadCalculator(risk_aversion=risk_aversion)
            
            self.logger.info("Spread calculator initialized successfully")
            return calculator
            
        except Exception as e:
            self.logger.error(f"Failed to initialize spread calculator: {str(e)}")
            raise
    
    def _initialize_inventory_manager(self):
        """
        Initialize the inventory manager.
        
        Returns:
            Initialized inventory manager
        """
        try:
            self.logger.info("Initializing inventory manager")
            
            # Import here to avoid circular imports
            from src.strategy.spread_calculator import InventoryManager
            
            # Get parameters from config
            target_inventory = self.config["strategy_params"]["general"]["target_inventory"]
            inventory_aversion = self.config["strategy_params"]["general"]["inventory_aversion"]
            
            # Initialize manager
            manager = InventoryManager(
                target_inventory=target_inventory,
                inventory_aversion=inventory_aversion
            )
            
            self.logger.info("Inventory manager initialized successfully")
            return manager
            
        except Exception as e:
            self.logger.error(f"Failed to initialize inventory manager: {str(e)}")
            raise
    
    def _initialize_informed_estimator(self):
        """
        Initialize the informed trading estimator.
        
        Returns:
            Initialized informed trading estimator
        """
        try:
            self.logger.info("Initializing informed trading estimator")
            
            # Import here to avoid circular imports
            from src.regime_detection.bayesian_estimator import BayesianInformedTradingEstimator
            
            # Get parameters from config
            prior_alpha = self.config["model_params"]["bayesian_estimator"]["prior_alpha"]
            prior_confidence = self.config["model_params"]["bayesian_estimator"]["prior_confidence"]
            
            # Initialize estimator
            estimator = BayesianInformedTradingEstimator(
                prior_alpha=prior_alpha,
                prior_confidence=prior_confidence
            )
            
            self.logger.info("Informed trading estimator initialized successfully")
            return estimator
            
        except Exception as e:
            self.logger.error(f"Failed to initialize informed trading estimator: {str(e)}")
            raise
    
    def _compute_current_features(self) -> np.ndarray:
        """
        Compute features from the current market data buffer.
        
        Returns:
            np.ndarray: Computed features
        """
        try:
            # This would be a more complex implementation in production
            # For this prototype, we'll assume the buffer already contains feature vectors
            
            # Get the most recent data points to form a sequence
            seq_length = self.config["model_params"]["autoencoder"]["seq_length"]
            if len(self.market_data_buffer) < seq_length:
                raise ValueError(f"Not enough data in buffer. Need {seq_length}, have {len(self.market_data_buffer)}")
            
            # Extract the most recent sequence
            recent_data = self.market_data_buffer[-seq_length:]
            
            # Convert to numpy array
            features = np.array(recent_data)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to compute features: {str(e)}")
            raise
    
    def _get_feature_sequence(self, features: np.ndarray) -> np.ndarray:
        """
        Prepare a feature sequence for the autoencoder.
        
        Args:
            features: Raw features
            
        Returns:
            np.ndarray: Prepared feature sequence
        """
        # Scale features if they're not already scaled
        if self.scaler is not None:
            scaled_features = self.scaler.transform(features)
        else:
            scaled_features = features
        
        # Reshape if needed
        if len(scaled_features.shape) == 2:
            # Already in correct shape (seq_length, input_channels)
            return scaled_features
        elif len(scaled_features.shape) == 1:
            # Single feature vector, reshape to (1, input_channels)
            return scaled_features.reshape(1, -1)
        else:
            raise ValueError(f"Unexpected feature shape: {scaled_features.shape}")
    
    def _map_regime_to_params(self, regime_label: int) -> Dict[str, Any]:
        """
        Map regime label to regime-specific parameters.
        
        Args:
            regime_label: Regime label from classifier
            
        Returns:
            Dict[str, Any]: Regime-specific parameters
        """
        # Map numeric label to regime name
        regime_names = ["normal", "volatile", "trending", "informed"]
        if regime_label < 0 or regime_label >= len(regime_names):
            self.logger.warning(f"Invalid regime label: {regime_label}, defaulting to 'normal'")
            regime_name = "normal"
        else:
            regime_name = regime_names[regime_label]
        
        # Get parameters for this regime
        if regime_name in self.regime_params:
            return self.regime_params[regime_name]
        else:
            self.logger.warning(f"No parameters found for regime: {regime_name}, using default")
            return self.regime_params["normal"]
    
    def on_market_data(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process incoming market data and update quotes.
        
        Args:
            market_data: Dictionary with latest market data
            
        Returns:
            Optional[Dict[str, Any]]: New quotes or None if no update needed
        """
        try:
            # Extract relevant data
            timestamp = market_data.get("timestamp", time.time())
            mid_price = market_data.get("mid_price")
            
            if mid_price is None:
                # Try to calculate mid price from bid/ask
                bid_price = market_data.get("bid_price")
                ask_price = market_data.get("ask_price")
                if bid_price is not None and ask_price is not None:
                    mid_price = (bid_price + ask_price) / 2
                else:
                    # Try to use last price
                    mid_price = market_data.get("last_price")
                    
                    if mid_price is None:
                        self.logger.error("Cannot determine mid price from market data")
                        return None
            
            # Extract or estimate volatility
            volatility = market_data.get("volatility")
            if volatility is None:
                # Use a default or calculate from recent price history
                volatility = 0.15  # Default annualized volatility
            
            # Update market data buffer
            # In a real implementation, this would compute features from raw data
            # For this prototype, we'll assume market_data already contains feature vector
            feature_vector = market_data.get("features")
            if feature_vector is not None:
                self.market_data_buffer.append(feature_vector)
            
            # Check if we have enough data
            seq_length = self.config["model_params"]["autoencoder"]["seq_length"]
            if len(self.market_data_buffer) < seq_length:
                self.logger.debug(f"Not enough data yet: {len(self.market_data_buffer)}/{seq_length}")
                return None
            
            # Compute features
            features = self._compute_current_features()
            
            # Prepare sequence for regime detection
            sequence = self._get_feature_sequence(features)
            
            # Detect regime
            regime_label, regime_prob = self.regime_classifier.classify(sequence)
            regime_params = self._map_regime_to_params(regime_label)
            
            # Update regime tracking
            self.current_regime = list(self.regime_params.keys())[regime_label]
            self.current_regime_prob = regime_prob
            
            # Record regime
            self.regime_history.append({
                "timestamp": timestamp,
                "regime": self.current_regime,
                "probability": regime_prob
            })
            
            # Update informed trading probability
            if len(self.order_history) > 0 and len(self.return_history) > 0:
                window_size = self.config["model_params"]["bayesian_estimator"]["update_window"]
                self.current_informed_prob = self.informed_estimator.update(
                    np.array(self.order_history[-window_size:]),
                    np.array(self.return_history[-window_size:]),
                    window_size=window_size
                )
            
            # Calculate optimal quotes
            bid_price, ask_price = self.spread_calculator.calculate_optimal_quotes(
                mid_price=mid_price,
                volatility=volatility,
                informed_trading_prob=self.current_informed_prob,
                inventory=self.current_inventory,
                target_inventory=self.target_inventory,
                inventory_aversion=regime_params["inventory_aversion"],
                max_inventory=self.max_position,
                regime_params=regime_params,
                tick_size=self.tick_size
            )
            
            # Determine order sizes
            base_size = self.config["strategy_params"]["general"]["order_size_base"]
            size_multiplier = regime_params["order_size_multiplier"]
            bid_size = round(base_size * size_multiplier)
            ask_size = round(base_size * size_multiplier)
            
            # Record quotes
            self.quotes.append({
                "timestamp": timestamp,
                "bid_price": bid_price,
                "ask_price": ask_price,
                "bid_size": bid_size,
                "ask_size": ask_size,
                "mid_price": mid_price,
                "regime": self.current_regime,
                "informed_prob": self.current_informed_prob,
                "inventory": self.current_inventory
            })
            
            # Return new quotes
            return {
                "timestamp": timestamp,
                "bid_price": bid_price,
                "ask_price": ask_price,
                "bid_size": bid_size,
                "ask_size": ask_size,
                "regime": self.current_regime,
                "regime_probability": regime_prob,
                "informed_probability": self.current_informed_prob
            }
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {str(e)}")
            return None
    
    def on_trade_execution(self, trade: Dict[str, Any]) -> None:
        """
        Process a trade execution.
        
        Args:
            trade: Dictionary with trade details
        """
        try:
            # Extract trade details
            timestamp = trade.get("timestamp", time.time())
            price = trade.get("price")
            size = trade.get("size")
            side = trade.get("side")  # "buy" or "sell" from market maker's perspective
            
            if price is None or size is None or side is None:
                self.logger.error("Missing required trade details")
                return
            
            # Update inventory
            if side == "buy":
                self.current_inventory += size
                order_direction = 1
            elif side == "sell":
                self.current_inventory -= size
                order_direction = -1
            else:
                self.logger.error(f"Invalid trade side: {side}")
                return
            
            # Record trade
            self.trades.append({
                "timestamp": timestamp,
                "price": price,
                "size": size,
                "side": side,
                "inventory": self.current_inventory
            })
            
            # Update order history for informed trading estimation
            self.order_history.append(order_direction)
            
            # We'll update return history when we observe the subsequent price change
            # For now, add a placeholder
            self.return_history.append(0.0)
            
            # Check if we need to hedge
            max_inventory = self.inventory_limits.get(self.current_regime, self.max_position)
            if self.inventory_manager.should_hedge(
                self.current_inventory,
                self.target_inventory,
                max_inventory
            ):
                hedge_size = self.inventory_manager.calculate_hedge_size(
                    self.current_inventory,
                    self.target_inventory
                )
                
                self.logger.info(f"Hedge recommended: size={hedge_size:.2f}")
                
                # In a real implementation, this would trigger a hedging order
                # For this prototype, we'll just log it
            
            self.logger.info(f"Processed trade: {side} {size} @ {price}, inventory={self.current_inventory}")
            
        except Exception as e:
            self.logger.error(f"Error processing trade execution: {str(e)}")
    
    def on_price_update(self, price_update: Dict[str, Any]) -> None:
        """
        Process a price update for PnL calculation and return history.
        
        Args:
            price_update: Dictionary with price update details
        """
        try:
            # Extract price details
            timestamp = price_update.get("timestamp", time.time())
            mid_price = price_update.get("mid_price")
            
            if mid_price is None:
                # Try to calculate mid price from bid/ask
                bid_price = price_update.get("bid_price")
                ask_price = price_update.get("ask_price")
                if bid_price is not None and ask_price is not None:
                    mid_price = (bid_price + ask_price) / 2
                else:
                    # Try to use last price
                    mid_price = price_update.get("last_price")
                    
                    if mid_price is None:
                        self.logger.error("Cannot determine mid price from price update")
                        return
            
            # Calculate mark-to-market PnL
            if self.trades:
                # Get previous price for PnL calculation
                prev_price = self.pnl_history[-1]["price"] if self.pnl_history else self.trades[0]["price"]
                
                # Calculate PnL
                price_change = mid_price - prev_price
                inventory_pnl = self.current_inventory * price_change * self.contract_multiplier
                
                # Record PnL
                self.pnl_history.append({
                    "timestamp": timestamp,
                    "price": mid_price,
                    "inventory": self.current_inventory,
                    "pnl_change": inventory_pnl,
                    "cumulative_pnl": self.pnl_history[-1]["cumulative_pnl"] + inventory_pnl if self.pnl_history else inventory_pnl
                })
                
                # Update return history for informed trading estimation
                if len(self.return_history) > 0:
                    # Calculate log return
                    log_return = np.log(mid_price / prev_price)
                    # Update the most recent return (which was a placeholder)
                    self.return_history[-1] = log_return
            
            self.logger.debug(f"Processed price update: mid_price={mid_price}")
            
        except Exception as e:
            self.logger.error(f"Error processing price update: {str(e)}")
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current state of the market maker.
        
        Returns:
            Dict[str, Any]: Current state
        """
        return {
            "timestamp": datetime.now().timestamp(),
            "inventory": self.current_inventory,
            "regime": self.current_regime,
            "regime_probability": self.current_regime_prob,
            "informed_probability": self.current_informed_prob,
            "pnl": self.pnl_history[-1]["cumulative_pnl"] if self.pnl_history else 0.0,
            "num_trades": len(self.trades),
            "num_quotes": len(self.quotes)
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics.
        
        Returns:
            Dict[str, Any]: Performance metrics
        """
        try:
            metrics = {
                "total_trades": len(self.trades),
                "total_volume": sum(t["size"] for t in self.trades),
                "final_inventory": self.current_inventory,
                "pnl": self.pnl_history[-1]["cumulative_pnl"] if self.pnl_history else 0.0
            }
            
            # Calculate additional metrics if we have trades
            if self.trades:
                # Calculate average spread
                spreads = [(q["ask_price"] - q["bid_price"]) for q in self.quotes]
                metrics["avg_spread"] = sum(spreads) / len(spreads)
                
                # Calculate spread capture
                buy_trades = [t for t in self.trades if t["side"] == "buy"]
                sell_trades = [t for t in self.trades if t["side"] == "sell"]
                
                if buy_trades and sell_trades:
                    avg_buy_price = sum(t["price"] * t["size"] for t in buy_trades) / sum(t["size"] for t in buy_trades)
                    avg_sell_price = sum(t["price"] * t["size"] for t in sell_trades) / sum(t["size"] for t in sell_trades)
                    metrics["spread_capture"] = avg_sell_price - avg_buy_price
                
                # Calculate regime distribution
                regime_counts = {}
                for r in self.regime_history:
                    regime = r["regime"]
                    regime_counts[regime] = regime_counts.get(regime, 0) + 1
                
                total_regimes = len(self.regime_history)
                metrics["regime_distribution"] = {
                    regime: count / total_regimes
                    for regime, count in regime_counts.items()
                }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return {"error": str(e)}
    
    def save_results(self, base_path: str) -> None:
        """
        Save trading results to files.
        
        Args:
            base_path: Base path for saving files
        """
        try:
            import pandas as pd
            import os
            
            # Create directory if it doesn't exist
            os.makedirs(base_path, exist_ok=True)
            
            # Save trades
            if self.trades:
                trades_df = pd.DataFrame(self.trades)
                trades_df.to_csv(f"{base_path}/trades.csv", index=False)
            
            # Save quotes
            if self.quotes:
                quotes_df = pd.DataFrame(self.quotes)
                quotes_df.to_csv(f"{base_path}/quotes.csv", index=False)
            
            # Save PnL history
            if self.pnl_history:
                pnl_df = pd.DataFrame(self.pnl_history)
                pnl_df.to_csv(f"{base_path}/pnl.csv", index=False)
            
            # Save regime history
            if self.regime_history:
                regime_df = pd.DataFrame(self.regime_history)
                regime_df.to_csv(f"{base_path}/regimes.csv", index=False)
            
            # Save performance metrics
            metrics = self.get_performance_metrics()
            with open(f"{base_path}/metrics.txt", "w") as f:
                for key, value in metrics.items():
                    f.write(f"{key}: {value}\n")
            
            self.logger.info(f"Results saved to {base_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")


# Example usage
if __name__ == "__main__":
    import yaml
    import json
    
    # Load configuration
    with open("/home/ubuntu/adaptive_market_making_implementation/config/strategy_params.yaml", 'r') as f:
        strategy_config = yaml.safe_load(f)
    
    with open("/home/ubuntu/adaptive_market_making_implementation/config/model_params.yaml", 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Combine configs
    config = {
        "strategy_params": strategy_config,
        "model_params": model_config,
        "model_paths": {
            "autoencoder": "/home/ubuntu/adaptive_market_making_implementation/models/autoencoder_final.pth",
            "gmm": "/home/ubuntu/adaptive_market_making_implementation/models/gmm_regime_model.pkl",
            "scaler": "/home/ubuntu/adaptive_market_making_implementation/models/feature_scaler.pkl"
        },
        "trading": {
            "tick_size": 0.25,
            "contract_multiplier": 50,
            "max_position": 100,
            "target_inventory": 0
        },
        "regime_specific_params": strategy_config["regime_strategies"]
    }
    
    # Initialize market maker
    market_maker = AdaptiveMarketMaker(config)
    
    # Simulate some market data and trades
    for i in range(100):
        # Simulate market data
        mid_price = 4500.0 + np.sin(i / 10) * 5.0
        market_data = {
            "timestamp": time.time(),
            "mid_price": mid_price,
            "bid_price": mid_price - 0.5,
            "ask_price": mid_price + 0.5,
            "volatility": 0.15,
            "features": np.random.randn(25)  # Random feature vector
        }
        
        # Process market data
        quotes = market_maker.on_market_data(market_data)
        
        # Simulate some trades (randomly)
        if quotes and np.random.random() < 0.2:
            # Decide if it's a buy or sell
            if np.random.random() < 0.5:
                side = "buy"
                price = quotes["bid_price"]
            else:
                side = "sell"
                price = quotes["ask_price"]
            
            # Create trade
            trade = {
                "timestamp": time.time(),
                "price": price,
                "size": np.random.randint(1, 5),
                "side": side
            }
            
            # Process trade
            market_maker.on_trade_execution(trade)
        
        # Update price for PnL calculation
        market_maker.on_price_update(market_data)
    
    # Print final state
    print(json.dumps(market_maker.get_current_state(), indent=2))
    
    # Print performance metrics
    print(json.dumps(market_maker.get_performance_metrics(), indent=2))
    
    # Save results
    market_maker.save_results("/home/ubuntu/adaptive_market_making_implementation/results")
