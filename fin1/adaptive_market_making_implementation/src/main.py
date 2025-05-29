"""
Main entry point for the adaptive market making strategy.
This module orchestrates the entire system.
"""

import logging
import argparse
import yaml
import os
import time
import signal
import sys
from typing import Dict, List, Optional, Union, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdaptiveMarketMakingSystem:
    """
    Main system class for the adaptive market making strategy.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the system.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__ + '.AdaptiveMarketMakingSystem')
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.data_collector = None
        self.data_consumer = None
        self.feature_engineering = None
        self.regime_detector = None
        self.market_maker = None
        self.execution_engine = None
        
        # State variables
        self.is_running = False
        
        self.logger.info("Initialized adaptive market making system")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dict[str, Any]: Loaded configuration
        """
        try:
            self.logger.info(f"Loading configuration from {config_path}")
            
            # Check file exists
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            # Load configuration
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.logger.info("Configuration loaded successfully")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def initialize(self) -> None:
        """
        Initialize all system components.
        """
        try:
            self.logger.info("Initializing system components")
            
            # Initialize data collection
            self._initialize_data_collection()
            
            # Initialize data consumption
            self._initialize_data_consumption()
            
            # Initialize feature engineering
            self._initialize_feature_engineering()
            
            # Initialize regime detection
            self._initialize_regime_detection()
            
            # Initialize market maker
            self._initialize_market_maker()
            
            # Initialize execution engine
            self._initialize_execution_engine()
            
            self.logger.info("System components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {str(e)}")
            raise
    
    def _initialize_data_collection(self) -> None:
        """
        Initialize data collection component.
        """
        try:
            self.logger.info("Initializing data collection")
            
            # Import data collection module
            from src.data_ingestion.market_data_ingestion import MarketDataCollector
            
            # Get data collection configuration
            data_collection_config = self.config.get("data_collection", {})
            
            # Initialize collector
            self.data_collector = MarketDataCollector(data_collection_config)
            
            self.logger.info("Data collection initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data collection: {str(e)}")
            raise
    
    def _initialize_data_consumption(self) -> None:
        """
        Initialize data consumption component.
        """
        try:
            self.logger.info("Initializing data consumption")
            
            # Import data consumption module
            from src.data_ingestion.market_data_ingestion import MarketDataConsumer
            
            # Get data consumption configuration
            data_consumption_config = self.config.get("data_consumption", {})
            
            # Initialize consumer
            self.data_consumer = MarketDataConsumer(data_consumption_config)
            
            self.logger.info("Data consumption initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data consumption: {str(e)}")
            raise
    
    def _initialize_feature_engineering(self) -> None:
        """
        Initialize feature engineering component.
        """
        try:
            self.logger.info("Initializing feature engineering")
            
            # Import feature engineering module
            from src.feature_engineering.feature_engineering import FeatureEngineeringPipeline
            
            # Get feature engineering configuration
            feature_engineering_config = self.config.get("feature_engineering", {})
            
            # Initialize pipeline
            self.feature_engineering = FeatureEngineeringPipeline(feature_engineering_config)
            
            self.logger.info("Feature engineering initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize feature engineering: {str(e)}")
            raise
    
    def _initialize_regime_detection(self) -> None:
        """
        Initialize regime detection component.
        """
        try:
            self.logger.info("Initializing regime detection")
            
            # Import regime detection modules
            from src.regime_detection.autoencoder import RegimeClassifier
            from src.regime_detection.bayesian_estimator import BayesianInformedTradingEstimator
            
            # Get regime detection configuration
            regime_detection_config = self.config.get("regime_detection", {})
            
            # Load models
            autoencoder_path = regime_detection_config.get("autoencoder_path")
            gmm_path = regime_detection_config.get("gmm_path")
            scaler_path = regime_detection_config.get("scaler_path")
            
            # Initialize regime detector
            # In a real implementation, this would load the models and initialize the detector
            # For this prototype, we'll just log it
            self.logger.info(f"Would load models from: {autoencoder_path}, {gmm_path}, {scaler_path}")
            
            self.logger.info("Regime detection initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize regime detection: {str(e)}")
            raise
    
    def _initialize_market_maker(self) -> None:
        """
        Initialize market maker component.
        """
        try:
            self.logger.info("Initializing market maker")
            
            # Import market maker module
            from src.strategy.adaptive_market_maker import AdaptiveMarketMaker
            
            # Get market maker configuration
            market_maker_config = self.config.get("market_maker", {})
            
            # Initialize market maker
            self.market_maker = AdaptiveMarketMaker(market_maker_config)
            
            self.logger.info("Market maker initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize market maker: {str(e)}")
            raise
    
    def _initialize_execution_engine(self) -> None:
        """
        Initialize execution engine component.
        """
        try:
            self.logger.info("Initializing execution engine")
            
            # Import execution engine module
            from src.execution.executor import ExecutionEngine
            
            # Get execution engine configuration
            execution_config = self.config.get("execution", {})
            
            # Initialize execution engine
            self.execution_engine = ExecutionEngine(execution_config)
            
            self.logger.info("Execution engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize execution engine: {str(e)}")
            raise
    
    def start(self) -> None:
        """
        Start the system.
        """
        try:
            if self.is_running:
                self.logger.warning("System is already running")
                return
            
            self.logger.info("Starting system")
            
            # Set running flag
            self.is_running = True
            
            # Start data collection
            if self.data_collector:
                self.data_collector.start_collection()
            
            # Start data consumption
            if self.data_consumer:
                self.data_consumer.start_consumption()
            
            # Start execution engine
            if self.execution_engine:
                self.execution_engine.start()
            
            self.logger.info("System started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start system: {str(e)}")
            self.is_running = False
            raise
    
    def stop(self) -> None:
        """
        Stop the system.
        """
        try:
            if not self.is_running:
                self.logger.warning("System is not running")
                return
            
            self.logger.info("Stopping system")
            
            # Clear running flag
            self.is_running = False
            
            # Stop execution engine
            if self.execution_engine:
                self.execution_engine.stop()
            
            # Stop data consumption
            if self.data_consumer:
                self.data_consumer.stop_consumption()
            
            # Stop data collection
            if self.data_collector:
                self.data_collector.stop_collection()
            
            self.logger.info("System stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to stop system: {str(e)}")
            raise
    
    def run_backtest(self, data_path: str, output_path: str) -> Dict[str, Any]:
        """
        Run backtest on historical data.
        
        Args:
            data_path: Path to historical data
            output_path: Path to save backtest results
            
        Returns:
            Dict[str, Any]: Backtest results
        """
        try:
            self.logger.info(f"Running backtest with data from {data_path}")
            
            # Import backtest engine
            from src.backtesting.backtest_engine import BacktestEngine
            
            # Get backtest configuration
            backtest_config = self.config.get("backtest", {})
            
            # Initialize backtest engine
            backtest_engine = BacktestEngine(backtest_config)
            
            # Load historical data
            data = backtest_engine.load_historical_data(data_path)
            
            # Preprocess data
            processed_data = backtest_engine.preprocess_data(data)
            
            # Run backtest
            results = backtest_engine.run_backtest(processed_data)
            
            # Save results
            backtest_engine.save_results(output_path)
            
            # Plot results
            backtest_engine.plot_results(os.path.join(output_path, "backtest_plot.png"))
            
            self.logger.info(f"Backtest completed successfully, results saved to {output_path}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to run backtest: {str(e)}")
            raise
    
    def run_simulation(self, output_path: str) -> Dict[str, Any]:
        """
        Run multi-agent simulation.
        
        Args:
            output_path: Path to save simulation results
            
        Returns:
            Dict[str, Any]: Simulation results
        """
        try:
            self.logger.info("Running multi-agent simulation")
            
            # Import simulation module
            from src.backtesting.multi_agent_simulation import MarketSimulator
            
            # Get simulation configuration
            simulation_config = self.config.get("simulation", {})
            
            # Initialize simulator
            simulator = MarketSimulator(simulation_config)
            
            # Run simulation
            results = simulator.run_simulation()
            
            # Save results
            simulator.save_results(output_path)
            
            # Plot results
            simulator.plot_results(os.path.join(output_path, "simulation_plot.png"))
            
            self.logger.info(f"Simulation completed successfully, results saved to {output_path}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to run simulation: {str(e)}")
            raise


def signal_handler(sig, frame):
    """
    Handle signals for graceful shutdown.
    
    Args:
        sig: Signal number
        frame: Current stack frame
    """
    logger.info(f"Received signal {sig}, shutting down...")
    
    # Access global system instance
    global system
    
    # Stop system if running
    if system and system.is_running:
        system.stop()
    
    # Exit
    sys.exit(0)


def main():
    """
    Main entry point.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Adaptive Market Making Strategy")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["live", "backtest", "simulation"], default="live", help="Operation mode")
    parser.add_argument("--data", type=str, help="Path to historical data for backtest mode")
    parser.add_argument("--output", type=str, default="results", help="Path to save results")
    args = parser.parse_args()
    
    try:
        # Initialize system
        global system
        system = AdaptiveMarketMakingSystem(args.config)
        system.initialize()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Run in specified mode
        if args.mode == "live":
            # Start system
            system.start()
            
            # Keep running until interrupted
            while system.is_running:
                time.sleep(1)
                
        elif args.mode == "backtest":
            # Check data path
            if not args.data:
                raise ValueError("Data path must be specified for backtest mode")
            
            # Create output directory
            os.makedirs(args.output, exist_ok=True)
            
            # Run backtest
            results = system.run_backtest(args.data, args.output)
            
            # Print summary
            print("\nBacktest Results:")
            print(f"Total trades: {results.get('total_trades', 0)}")
            print(f"Final PnL: {results.get('pnl', 0):.2f}")
            print(f"Sharpe ratio: {results.get('sharpe_ratio', 0):.2f}")
            
        elif args.mode == "simulation":
            # Create output directory
            os.makedirs(args.output, exist_ok=True)
            
            # Run simulation
            results = system.run_simulation(args.output)
            
            # Print summary
            print("\nSimulation Results:")
            print(f"Total trades: {results.get('total_trades', 0)}")
            print(f"Final price: {results.get('final_price', 0):.2f}")
            
            # Print market maker metrics
            mm_metrics = results.get("agent_metrics", {}).get("market_maker", {})
            if mm_metrics:
                print("\nMarket Maker Metrics:")
                print(f"Final inventory: {mm_metrics.get('final_inventory', 0)}")
                print(f"Final PnL: {mm_metrics.get('pnl', 0):.2f}")
                print(f"Number of trades: {mm_metrics.get('num_trades', 0)}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
