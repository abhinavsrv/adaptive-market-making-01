"""
Backtesting engine for the adaptive market making strategy.
This module provides functionality to backtest the strategy on historical data.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import os
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Engine for backtesting the adaptive market making strategy.
    """
    
    def __init__(self, config: Dict[str, Any], market_maker=None):
        """
        Initialize the backtest engine.
        
        Args:
            config: Dictionary containing configuration parameters
            market_maker: Optional pre-initialized market maker instance
        """
        self.config = config
        self.logger = logging.getLogger(__name__ + '.BacktestEngine')
        
        # Initialize market maker if not provided
        if market_maker is None:
            from src.strategy.adaptive_market_maker import AdaptiveMarketMaker
            self.market_maker = AdaptiveMarketMaker(config)
        else:
            self.market_maker = market_maker
        
        # Backtest parameters
        self.start_date = config["backtest"]["start_date"]
        self.end_date = config["backtest"]["end_date"]
        self.tick_size = config["trading"]["tick_size"]
        self.contract_multiplier = config["trading"]["contract_multiplier"]
        
        # Performance tracking
        self.trades = []
        self.quotes = []
        self.pnl_history = []
        self.metrics = {}
        
        self.logger.info("Initialized backtest engine")
    
    def load_historical_data(self, data_path: str) -> pd.DataFrame:
        """
        Load historical market data for backtesting.
        
        Args:
            data_path: Path to historical data file
            
        Returns:
            pd.DataFrame: Loaded historical data
        """
        try:
            self.logger.info(f"Loading historical data from {data_path}")
            
            # Determine file format from extension
            file_ext = os.path.splitext(data_path)[1].lower()
            
            if file_ext == '.csv':
                data = pd.read_csv(data_path)
            elif file_ext == '.parquet':
                data = pd.read_parquet(data_path)
            elif file_ext == '.feather':
                data = pd.read_feather(data_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Ensure timestamp column exists and is in datetime format
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            else:
                raise ValueError("Data must contain a 'timestamp' column")
            
            # Filter by date range if specified
            if self.start_date:
                start_date = pd.to_datetime(self.start_date)
                data = data[data['timestamp'] >= start_date]
            
            if self.end_date:
                end_date = pd.to_datetime(self.end_date)
                data = data[data['timestamp'] <= end_date]
            
            # Sort by timestamp
            data = data.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.info(f"Loaded {len(data)} data points from {data['timestamp'].min()} to {data['timestamp'].max()}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load historical data: {str(e)}")
            raise
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess historical data for backtesting.
        
        Args:
            data: Raw historical data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        try:
            self.logger.info("Preprocessing historical data")
            
            # Check required columns
            required_columns = ['timestamp', 'price']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Data must contain columns: {required_columns}")
            
            # Calculate mid price if not available
            if 'mid_price' not in data.columns:
                if 'bid_price' in data.columns and 'ask_price' in data.columns:
                    data['mid_price'] = (data['bid_price'] + data['ask_price']) / 2
                else:
                    data['mid_price'] = data['price']
            
            # Calculate returns
            data['return'] = data['mid_price'].pct_change()
            
            # Calculate volatility (rolling 20-period standard deviation of returns)
            data['volatility'] = data['return'].rolling(window=20).std() * np.sqrt(252 * 6.5 * 3600 / 20)  # Annualized
            
            # Forward fill volatility
            data['volatility'] = data['volatility'].fillna(method='ffill')
            
            # Set initial volatility if still NaN
            data['volatility'] = data['volatility'].fillna(0.15)  # Default annualized volatility
            
            # Compute features if not available
            if 'features' not in data.columns:
                # In a real implementation, this would compute features from raw data
                # For this prototype, we'll create dummy features
                
                # Get feature dimension from config
                input_channels = self.config["model_params"]["autoencoder"]["input_channels"]
                
                # Create dummy features
                features = []
                for i in range(len(data)):
                    # Create a feature vector based on price, volatility, etc.
                    # This is a simplified placeholder
                    row = data.iloc[i]
                    price = row['mid_price']
                    vol = row['volatility']
                    ret = row['return'] if not pd.isna(row['return']) else 0
                    
                    # Base feature on these values plus some noise
                    feature = np.zeros(input_channels)
                    feature[0] = price / 1000  # Normalized price
                    feature[1] = vol * 10  # Scaled volatility
                    feature[2] = ret * 100 if not pd.isna(ret) else 0  # Scaled return
                    
                    # Add some autocorrelated noise for the rest
                    if i > 0:
                        prev_feature = features[-1][3:]
                        noise = prev_feature * 0.8 + np.random.randn(input_channels - 3) * 0.2
                    else:
                        noise = np.random.randn(input_channels - 3)
                    
                    feature[3:] = noise
                    features.append(feature)
                
                # Add features to dataframe
                data['features'] = features
            
            self.logger.info("Data preprocessing complete")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to preprocess data: {str(e)}")
            raise
    
    def run_backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run backtest on historical data.
        
        Args:
            data: Preprocessed historical data
            
        Returns:
            Dict[str, Any]: Backtest results
        """
        try:
            self.logger.info("Starting backtest")
            
            # Reset market maker state
            # In a real implementation, this would reset the market maker
            # For this prototype, we'll just create a new instance
            from src.strategy.adaptive_market_maker import AdaptiveMarketMaker
            self.market_maker = AdaptiveMarketMaker(self.config)
            
            # Initialize tracking variables
            self.trades = []
            self.quotes = []
            self.pnl_history = []
            
            # Track execution time
            start_time = time.time()
            
            # Process each data point
            for i, row in enumerate(data.itertuples()):
                # Convert row to dictionary for market maker
                market_data = {
                    "timestamp": row.timestamp.timestamp(),
                    "mid_price": row.mid_price,
                    "volatility": row.volatility,
                    "features": row.features
                }
                
                # Add bid/ask if available
                if hasattr(row, 'bid_price') and hasattr(row, 'ask_price'):
                    market_data["bid_price"] = row.bid_price
                    market_data["ask_price"] = row.ask_price
                
                # Process market data
                quotes = self.market_maker.on_market_data(market_data)
                
                # If quotes were generated, check for executions
                if quotes:
                    # In a real backtest, this would use the next tick's data to determine executions
                    # For this prototype, we'll use a simple model
                    
                    # Only consider executions after some initial data points
                    if i > 20:
                        # Simulate executions based on market data
                        self._simulate_executions(quotes, market_data)
                
                # Update price for PnL calculation
                self.market_maker.on_price_update(market_data)
                
                # Log progress periodically
                if i % 1000 == 0:
                    elapsed = time.time() - start_time
                    self.logger.info(f"Processed {i}/{len(data)} data points ({i/len(data)*100:.1f}%) in {elapsed:.1f}s")
            
            # Calculate final metrics
            self.metrics = self.market_maker.get_performance_metrics()
            
            # Add additional backtest-specific metrics
            self.metrics["backtest_duration_seconds"] = time.time() - start_time
            self.metrics["data_points_processed"] = len(data)
            
            # Calculate Sharpe ratio if we have PnL history
            if self.market_maker.pnl_history:
                pnl_df = pd.DataFrame(self.market_maker.pnl_history)
                pnl_df['timestamp'] = pd.to_datetime(pnl_df['timestamp'], unit='s')
                pnl_df = pnl_df.set_index('timestamp')
                
                # Calculate daily returns
                pnl_df['daily_pnl'] = pnl_df['pnl_change'].resample('D').sum()
                
                # Calculate Sharpe ratio (annualized)
                daily_returns = pnl_df['daily_pnl'].dropna()
                if len(daily_returns) > 1:
                    sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
                    self.metrics["sharpe_ratio"] = sharpe
            
            self.logger.info(f"Backtest completed with {self.metrics.get('total_trades', 0)} trades")
            
            return self.metrics
            
        except Exception as e:
            self.logger.error(f"Failed to run backtest: {str(e)}")
            raise
    
    def _simulate_executions(self, quotes: Dict[str, Any], market_data: Dict[str, Any]) -> None:
        """
        Simulate trade executions based on quotes and market data.
        
        Args:
            quotes: Quote data from market maker
            market_data: Current market data
        """
        # Extract quote details
        bid_price = quotes["bid_price"]
        ask_price = quotes["ask_price"]
        bid_size = quotes["bid_size"]
        ask_size = quotes["ask_size"]
        
        # Extract market data
        mid_price = market_data["mid_price"]
        timestamp = market_data["timestamp"]
        
        # Simple execution model:
        # - If mid price is below bid, someone might hit our bid (we buy)
        # - If mid price is above ask, someone might lift our offer (we sell)
        
        # Add some randomness to make it more realistic
        execution_threshold = 0.3  # Probability of execution when price crosses
        
        # Check for bid execution (we buy)
        if mid_price <= bid_price and np.random.random() < execution_threshold:
            # Create trade
            trade = {
                "timestamp": timestamp,
                "price": bid_price,
                "size": np.random.randint(1, bid_size + 1),  # Random size up to bid_size
                "side": "buy"  # We buy
            }
            
            # Process trade
            self.market_maker.on_trade_execution(trade)
        
        # Check for ask execution (we sell)
        if mid_price >= ask_price and np.random.random() < execution_threshold:
            # Create trade
            trade = {
                "timestamp": timestamp,
                "price": ask_price,
                "size": np.random.randint(1, ask_size + 1),  # Random size up to ask_size
                "side": "sell"  # We sell
            }
            
            # Process trade
            self.market_maker.on_trade_execution(trade)
    
    def save_results(self, base_path: str) -> None:
        """
        Save backtest results to files.
        
        Args:
            base_path: Base path for saving files
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(base_path, exist_ok=True)
            
            # Save market maker results
            self.market_maker.save_results(base_path)
            
            # Save backtest metrics
            with open(f"{base_path}/backtest_metrics.json", "w") as f:
                json.dump(self.metrics, f, indent=2)
            
            self.logger.info(f"Backtest results saved to {base_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save backtest results: {str(e)}")
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """
        Plot backtest results.
        
        Args:
            save_path: Optional path to save plots
        """
        try:
            # Check if we have PnL history
            if not self.market_maker.pnl_history:
                self.logger.warning("No PnL history available for plotting")
                return
            
            # Convert to DataFrame
            pnl_df = pd.DataFrame(self.market_maker.pnl_history)
            pnl_df['timestamp'] = pd.to_datetime(pnl_df['timestamp'], unit='s')
            pnl_df = pnl_df.set_index('timestamp')
            
            # Create figure with subplots
            fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
            
            # Plot cumulative PnL
            axes[0].plot(pnl_df.index, pnl_df['cumulative_pnl'], 'b-')
            axes[0].set_title('Cumulative PnL')
            axes[0].set_ylabel('PnL')
            axes[0].grid(True)
            
            # Plot inventory
            axes[1].plot(pnl_df.index, pnl_df['inventory'], 'g-')
            axes[1].set_title('Inventory')
            axes[1].set_ylabel('Position')
            axes[1].grid(True)
            
            # Plot price
            axes[2].plot(pnl_df.index, pnl_df['price'], 'k-')
            axes[2].set_title('Price')
            axes[2].set_ylabel('Price')
            axes[2].grid(True)
            
            # Format x-axis
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save or show
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Results plot saved to {save_path}")
            else:
                plt.show()
            
        except Exception as e:
            self.logger.error(f"Failed to plot results: {str(e)}")


# Example usage
if __name__ == "__main__":
    import yaml
    
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
        "backtest": {
            "start_date": "2023-01-01",
            "end_date": "2023-12-31"
        },
        "regime_specific_params": strategy_config["regime_strategies"]
    }
    
    # Initialize backtest engine
    backtest_engine = BacktestEngine(config)
    
    # Generate synthetic data for testing
    # In a real implementation, this would load actual historical data
    dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq="1min")
    prices = 4500 + np.cumsum(np.random.normal(0, 1, len(dates)) * 0.1)
    
    data = pd.DataFrame({
        "timestamp": dates,
        "price": prices,
        "bid_price": prices - 0.5,
        "ask_price": prices + 0.5
    })
    
    # Preprocess data
    processed_data = backtest_engine.preprocess_data(data)
    
    # Run backtest
    results = backtest_engine.run_backtest(processed_data)
    
    # Print results
    print(json.dumps(results, indent=2))
    
    # Save results
    backtest_engine.save_results("/home/ubuntu/adaptive_market_making_implementation/results/backtest")
    
    # Plot results
    backtest_engine.plot_results("/home/ubuntu/adaptive_market_making_implementation/results/backtest/performance_plot.png")
