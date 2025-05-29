"""
Feature engineering pipeline for the adaptive market making strategy.
This module transforms raw market data into features for the regime detection model.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta

from pyflink.datastream import StreamExecutionEnvironment, TimeCharacteristic
from pyflink.datastream.connectors import FlinkKafkaConsumer, FlinkKafkaProducer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
from pyflink.datastream.functions import MapFunction, ProcessFunction
from pyflink.common import WatermarkStrategy, Time
from pyflink.datastream.window import TumblingEventTimeWindows, TimeWindow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineeringPipeline:
    """
    Feature engineering pipeline using Apache Flink for stream processing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature engineering pipeline.
        
        Args:
            config: Configuration dictionary with pipeline parameters
        """
        self.config = config
        self.kafka_config = config['kafka']
        self.flink_config = config['flink']
        self.env = None
        self.logger = logging.getLogger(__name__ + '.FeatureEngineeringPipeline')
        
        self.logger.info("Initializing feature engineering pipeline")
        
    def initialize_flink_environment(self) -> None:
        """
        Initialize the Flink execution environment.
        """
        try:
            self.logger.info("Setting up Flink execution environment")
            
            # Create execution environment
            self.env = StreamExecutionEnvironment.get_execution_environment()
            
            # Configure environment
            self.env.set_stream_time_characteristic(TimeCharacteristic.EventTime)
            self.env.set_parallelism(self.flink_config['parallelism']['default'])
            
            # Enable checkpointing
            self.env.enable_checkpointing(self.flink_config['checkpointing']['interval'])
            checkpoint_config = self.env.get_checkpoint_config()
            checkpoint_config.set_checkpoint_timeout(self.flink_config['checkpointing']['timeout'])
            checkpoint_config.set_min_pause_between_checkpoints(
                self.flink_config['checkpointing']['min_pause_between_checkpoints']
            )
            checkpoint_config.set_max_concurrent_checkpoints(
                self.flink_config['checkpointing']['max_concurrent_checkpoints']
            )
            
            self.logger.info("Flink execution environment initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Flink environment: {str(e)}")
            raise
    
    def create_kafka_consumer(self, topic: str, group_id: str) -> FlinkKafkaConsumer:
        """
        Create a Flink Kafka consumer.
        
        Args:
            topic: Kafka topic to consume from
            group_id: Consumer group ID
            
        Returns:
            FlinkKafkaConsumer: Configured Kafka consumer
        """
        try:
            self.logger.info(f"Creating Kafka consumer for topic: {topic}")
            
            properties = {
                'bootstrap.servers': ','.join(self.kafka_config['bootstrap_servers']),
                'group.id': group_id,
                'auto.offset.reset': self.kafka_config['auto_offset_reset'],
                'enable.auto.commit': str(self.kafka_config['enable_auto_commit']).lower(),
                'auto.commit.interval.ms': str(self.kafka_config['auto_commit_interval_ms']),
                'session.timeout.ms': str(self.kafka_config['session_timeout_ms'])
            }
            
            consumer = FlinkKafkaConsumer(
                topic,
                SimpleStringSchema(),
                properties
            )
            
            # Configure watermark strategy for event time processing
            consumer.set_start_from_latest()
            
            # Use a timestamp extractor and watermark generator
            watermark_strategy = (
                WatermarkStrategy
                .for_bounded_out_of_orderness(Time.milliseconds(200))
                .with_timestamp_assigner(self._timestamp_assigner)
            )
            consumer.assign_timestamps_and_watermarks(watermark_strategy)
            
            return consumer
            
        except Exception as e:
            self.logger.error(f"Failed to create Kafka consumer: {str(e)}")
            raise
    
    def create_kafka_producer(self, topic: str) -> FlinkKafkaProducer:
        """
        Create a Flink Kafka producer.
        
        Args:
            topic: Kafka topic to produce to
            
        Returns:
            FlinkKafkaProducer: Configured Kafka producer
        """
        try:
            self.logger.info(f"Creating Kafka producer for topic: {topic}")
            
            properties = {
                'bootstrap.servers': ','.join(self.kafka_config['bootstrap_servers']),
                'transaction.timeout.ms': '900000'
            }
            
            producer = FlinkKafkaProducer(
                topic,
                SimpleStringSchema(),
                properties
            )
            
            return producer
            
        except Exception as e:
            self.logger.error(f"Failed to create Kafka producer: {str(e)}")
            raise
    
    def _timestamp_assigner(self, event, timestamp):
        """
        Extract timestamp from event for watermarking.
        
        Args:
            event: Event data
            timestamp: Previous timestamp
            
        Returns:
            int: Extracted timestamp in milliseconds
        """
        try:
            # Parse the event as JSON
            import json
            data = json.loads(event)
            
            # Extract timestamp field
            event_time = int(data.get('_timestamp', 0) * 1000)  # Convert to milliseconds
            return event_time
            
        except Exception:
            return timestamp
    
    def build_feature_engineering_pipeline(self) -> None:
        """
        Build the complete feature engineering pipeline.
        """
        try:
            self.logger.info("Building feature engineering pipeline")
            
            if not self.env:
                self.initialize_flink_environment()
            
            # Create Kafka consumers for raw data
            trades_consumer = self.create_kafka_consumer(
                self.kafka_config['topics']['raw_trades'],
                f"{self.kafka_config['consumer_group']}_trades"
            )
            
            orderbook_consumer = self.create_kafka_consumer(
                self.kafka_config['topics']['raw_orderbook'],
                f"{self.kafka_config['consumer_group']}_orderbook"
            )
            
            # Create data streams
            trades_stream = self.env.add_source(trades_consumer)
            orderbook_stream = self.env.add_source(orderbook_consumer)
            
            # Parse JSON
            trades_parsed = trades_stream.map(self._parse_json).returns(Types.MAP(Types.STRING(), Types.STRING()))
            orderbook_parsed = orderbook_stream.map(self._parse_json).returns(Types.MAP(Types.STRING(), Types.STRING()))
            
            # Clean data
            trades_cleaned = trades_parsed.map(self._clean_trade_data).returns(Types.MAP(Types.STRING(), Types.STRING()))
            orderbook_cleaned = orderbook_parsed.map(self._clean_orderbook_data).returns(Types.MAP(Types.STRING(), Types.STRING()))
            
            # Window data for feature calculation
            trades_windowed = trades_cleaned \
                .key_by(lambda x: x['symbol']) \
                .window(TumblingEventTimeWindows.of(Time.milliseconds(100))) \
                .apply(self._aggregate_trades)
                
            orderbook_windowed = orderbook_cleaned \
                .key_by(lambda x: x['symbol']) \
                .window(TumblingEventTimeWindows.of(Time.milliseconds(100))) \
                .apply(self._aggregate_orderbook)
            
            # Join trade and orderbook data
            # Note: In a real implementation, this would use CoProcessFunction or similar
            # For this prototype, we'll process them separately
            
            # Calculate features
            trade_features = trades_windowed.process(self._calculate_trade_features)
            orderbook_features = orderbook_windowed.process(self._calculate_orderbook_features)
            
            # Combine features
            # Note: In a real implementation, this would use proper stream joining
            # For this prototype, we'll process them separately
            
            # Create Kafka producer for processed features
            features_producer = self.create_kafka_producer(
                self.kafka_config['topics']['processed_features']
            )
            
            # Output features to Kafka
            trade_features.add_sink(features_producer)
            orderbook_features.add_sink(features_producer)
            
            self.logger.info("Feature engineering pipeline built successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to build feature engineering pipeline: {str(e)}")
            raise
    
    def _parse_json(self, json_str: str) -> Dict[str, Any]:
        """
        Parse JSON string to dictionary.
        
        Args:
            json_str: JSON string
            
        Returns:
            Dict[str, Any]: Parsed dictionary
        """
        import json
        return json.loads(json_str)
    
    def _clean_trade_data(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean raw trade data.
        
        Args:
            trade: Raw trade data
            
        Returns:
            Dict[str, Any]: Cleaned trade data
        """
        # Ensure required fields exist
        if not all(k in trade for k in ['symbol', 'timestamp', 'price', 'volume']):
            return None
        
        # Convert types
        trade['price'] = float(trade['price'])
        trade['volume'] = float(trade['volume'])
        trade['timestamp'] = float(trade['timestamp'])
        
        # Filter invalid values
        if trade['price'] <= 0 or trade['volume'] <= 0:
            return None
        
        return trade
    
    def _clean_orderbook_data(self, orderbook: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean raw order book data.
        
        Args:
            orderbook: Raw order book data
            
        Returns:
            Dict[str, Any]: Cleaned order book data
        """
        # Ensure required fields exist
        if not all(k in orderbook for k in ['symbol', 'timestamp', 'bid_price', 'ask_price', 'bid_size', 'ask_size']):
            return None
        
        # Convert types
        orderbook['bid_price'] = float(orderbook['bid_price'])
        orderbook['ask_price'] = float(orderbook['ask_price'])
        orderbook['bid_size'] = float(orderbook['bid_size'])
        orderbook['ask_size'] = float(orderbook['ask_size'])
        orderbook['timestamp'] = float(orderbook['timestamp'])
        
        # Filter invalid values
        if orderbook['bid_price'] <= 0 or orderbook['ask_price'] <= 0 or orderbook['bid_size'] <= 0 or orderbook['ask_size'] <= 0:
            return None
        
        # Ensure bid < ask
        if orderbook['bid_price'] >= orderbook['ask_price']:
            return None
        
        return orderbook
    
    def _aggregate_trades(self, key: str, window: TimeWindow, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate trades within a time window.
        
        Args:
            key: Key (symbol)
            window: Time window
            trades: List of trades in the window
            
        Returns:
            Dict[str, Any]: Aggregated trade data
        """
        if not trades:
            return None
        
        # Sort by timestamp
        sorted_trades = sorted(trades, key=lambda x: x['timestamp'])
        
        # Calculate VWAP
        total_volume = sum(t['volume'] for t in sorted_trades)
        vwap = sum(t['price'] * t['volume'] for t in sorted_trades) / total_volume if total_volume > 0 else 0
        
        # Calculate other metrics
        prices = [t['price'] for t in sorted_trades]
        volumes = [t['volume'] for t in sorted_trades]
        
        # Determine trade direction (simplified)
        buy_volume = sum(t['volume'] for t in sorted_trades if t.get('side') == 'buy')
        sell_volume = sum(t['volume'] for t in sorted_trades if t.get('side') == 'sell')
        unknown_volume = total_volume - buy_volume - sell_volume
        
        # Adjust volumes if direction is unknown
        if unknown_volume > 0:
            # Distribute unknown volume proportionally
            if buy_volume + sell_volume > 0:
                buy_ratio = buy_volume / (buy_volume + sell_volume)
                buy_volume += unknown_volume * buy_ratio
                sell_volume += unknown_volume * (1 - buy_ratio)
            else:
                # If all volume is unknown, assume equal distribution
                buy_volume = unknown_volume / 2
                sell_volume = unknown_volume / 2
        
        # Calculate order flow imbalance
        order_flow_imbalance = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
        
        return {
            'symbol': key,
            'timestamp': window.end,
            'window_start': window.start,
            'window_end': window.end,
            'type': 'trade_agg',
            'vwap': vwap,
            'min_price': min(prices),
            'max_price': max(prices),
            'open_price': prices[0],
            'close_price': prices[-1],
            'total_volume': total_volume,
            'num_trades': len(trades),
            'avg_trade_size': total_volume / len(trades),
            'order_flow_imbalance': order_flow_imbalance
        }
    
    def _aggregate_orderbook(self, key: str, window: TimeWindow, orderbooks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate order book data within a time window.
        
        Args:
            key: Key (symbol)
            window: Time window
            orderbooks: List of order book snapshots in the window
            
        Returns:
            Dict[str, Any]: Aggregated order book data
        """
        if not orderbooks:
            return None
        
        # Sort by timestamp
        sorted_obs = sorted(orderbooks, key=lambda x: x['timestamp'])
        
        # Calculate metrics
        spreads = [ob['ask_price'] - ob['bid_price'] for ob in sorted_obs]
        mid_prices = [(ob['ask_price'] + ob['bid_price']) / 2 for ob in sorted_obs]
        bid_sizes = [ob['bid_size'] for ob in sorted_obs]
        ask_sizes = [ob['ask_size'] for ob in sorted_obs]
        
        # Calculate book imbalance
        imbalances = [(ob['bid_size'] - ob['ask_size']) / (ob['bid_size'] + ob['ask_size']) 
                      if (ob['bid_size'] + ob['ask_size']) > 0 else 0 
                      for ob in sorted_obs]
        
        return {
            'symbol': key,
            'timestamp': window.end,
            'window_start': window.start,
            'window_end': window.end,
            'type': 'orderbook_agg',
            'avg_spread': sum(spreads) / len(spreads),
            'min_spread': min(spreads),
            'max_spread': max(spreads),
            'avg_mid_price': sum(mid_prices) / len(mid_prices),
            'avg_bid_size': sum(bid_sizes) / len(bid_sizes),
            'avg_ask_size': sum(ask_sizes) / len(ask_sizes),
            'avg_book_imbalance': sum(imbalances) / len(imbalances),
            'final_bid': sorted_obs[-1]['bid_price'],
            'final_ask': sorted_obs[-1]['ask_price'],
            'final_bid_size': sorted_obs[-1]['bid_size'],
            'final_ask_size': sorted_obs[-1]['ask_size']
        }
    
    def _calculate_trade_features(self, trade_agg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate features from aggregated trade data.
        
        Args:
            trade_agg: Aggregated trade data
            
        Returns:
            Dict[str, Any]: Trade features
        """
        # This would be a more complex implementation in production
        # For this prototype, we'll calculate some basic features
        
        features = {
            'symbol': trade_agg['symbol'],
            'timestamp': trade_agg['timestamp'],
            'type': 'features',
            'source': 'trade',
            'price_change': trade_agg['close_price'] - trade_agg['open_price'],
            'price_range': trade_agg['max_price'] - trade_agg['min_price'],
            'volume': trade_agg['total_volume'],
            'num_trades': trade_agg['num_trades'],
            'avg_trade_size': trade_agg['avg_trade_size'],
            'order_flow_imbalance': trade_agg['order_flow_imbalance']
        }
        
        return features
    
    def _calculate_orderbook_features(self, ob_agg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate features from aggregated order book data.
        
        Args:
            ob_agg: Aggregated order book data
            
        Returns:
            Dict[str, Any]: Order book features
        """
        # This would be a more complex implementation in production
        # For this prototype, we'll calculate some basic features
        
        features = {
            'symbol': ob_agg['symbol'],
            'timestamp': ob_agg['timestamp'],
            'type': 'features',
            'source': 'orderbook',
            'spread': ob_agg['avg_spread'],
            'mid_price': ob_agg['avg_mid_price'],
            'book_imbalance': ob_agg['avg_book_imbalance'],
            'liquidity_ratio': (ob_agg['avg_bid_size'] + ob_agg['avg_ask_size']) / ob_agg['avg_spread'] 
                              if ob_agg['avg_spread'] > 0 else 0
        }
        
        return features
    
    def execute(self) -> None:
        """
        Execute the feature engineering pipeline.
        """
        try:
            self.logger.info("Executing feature engineering pipeline")
            
            if not self.env:
                self.build_feature_engineering_pipeline()
            
            # Execute the pipeline
            self.env.execute("Adaptive Market Making Feature Engineering Pipeline")
            
        except Exception as e:
            self.logger.error(f"Failed to execute feature engineering pipeline: {str(e)}")
            raise


class FeatureProcessor:
    """
    Processor for computing features from market data.
    This is a non-Flink implementation for local processing and testing.
    """
    
    def __init__(self, window_size: int = 100, resample_freq: str = "100ms"):
        """
        Initialize the feature processor.
        
        Args:
            window_size: Rolling window size for feature calculation
            resample_freq: Frequency for resampling features
        """
        self.window_size = window_size
        self.resample_freq = resample_freq
        self.logger = logging.getLogger(__name__ + '.FeatureProcessor')
        
        self.logger.info(f"Initializing feature processor with window_size={window_size}, resample_freq={resample_freq}")
    
    def clean_raw_data(self, raw_ticks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw tick data.
        
        Args:
            raw_ticks_df: DataFrame with columns [timestamp, price, volume, side]
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        self.logger.info(f"Cleaning raw tick data with shape {raw_ticks_df.shape}")
        
        # Ensure correct data types
        raw_ticks_df["timestamp"] = pd.to_datetime(raw_ticks_df["timestamp"])
        raw_ticks_df["price"] = pd.to_numeric(raw_ticks_df["price"])
        raw_ticks_df["volume"] = pd.to_numeric(raw_ticks_df["volume"])
        
        # Sort by timestamp
        raw_ticks_df = raw_ticks_df.sort_values("timestamp").reset_index(drop=True)
        
        # Remove duplicate ticks
        raw_ticks_df = raw_ticks_df.drop_duplicates(subset=["timestamp", "price", "volume", "side"])
        
        # Filter out zero or negative prices/volumes
        raw_ticks_df = raw_ticks_df[(raw_ticks_df["price"] > 0) & (raw_ticks_df["volume"] > 0)]
        
        # Filter outliers based on price returns
        raw_ticks_df["log_return"] = np.log(raw_ticks_df["price"]).diff()
        mean_return = raw_ticks_df["log_return"].mean()
        std_return = raw_ticks_df["log_return"].std()
        
        # Remove ticks with returns > 5 standard deviations from the mean
        raw_ticks_df = raw_ticks_df[abs(raw_ticks_df["log_return"] - mean_return) < 5 * std_return]
        
        # Handle missing data (forward fill)
        raw_ticks_df = raw_ticks_df.ffill()
        
        self.logger.info(f"Cleaned tick data shape: {raw_ticks_df.shape}")
        
        return raw_ticks_df.drop(columns=["log_return"])
    
    def compute_features_from_ticks(self, cleaned_ticks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features from cleaned tick data.
        
        Args:
            cleaned_ticks_df: DataFrame of cleaned ticks
            
        Returns:
            pd.DataFrame: DataFrame of computed features
        """
        self.logger.info(f"Computing features from ticks with shape {cleaned_ticks_df.shape}")
        
        # Set timestamp as index
        cleaned_ticks_df = cleaned_ticks_df.set_index("timestamp")
        
        # Resample to desired frequency (e.g., 100ms)
        resampled_data = cleaned_ticks_df["price"].resample(self.resample_freq).last().ffill()
        volume_data = cleaned_ticks_df["volume"].resample(self.resample_freq).sum()
        
        # Calculate log returns
        log_returns = np.log(resampled_data).diff().fillna(0)
        
        # Calculate volatility features
        vol_5 = log_returns.rolling(window=5).std().fillna(0)
        vol_20 = log_returns.rolling(window=20).std().fillna(0)
        vol_50 = log_returns.rolling(window=50).std().fillna(0)
        
        # Calculate momentum features
        mom_5 = resampled_data.pct_change(periods=5).fillna(0)
        mom_20 = resampled_data.pct_change(periods=20).fillna(0)
        
        # Calculate order flow imbalance (requires bid/ask data, simplified here)
        # Assuming we have bid/ask prices and volumes available
        # bid_ask_spread = ask_price - bid_price
        # book_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        # Placeholder features:
        bid_ask_spread = vol_20 * 0.1  # Placeholder
        book_imbalance = np.random.randn(len(resampled_data)) * 0.1  # Placeholder
        
        # Combine features into a DataFrame
        features_df = pd.DataFrame({
            "log_return": log_returns,
            "vol_5": vol_5,
            "vol_20": vol_20,
            "vol_50": vol_50,
            "mom_5": mom_5,
            "mom_20": mom_20,
            "spread_norm": bid_ask_spread / (vol_20 + 1e-8),
            "imbalance": book_imbalance,
            "volume_norm": volume_data / (volume_data.rolling(window=self.window_size).mean() + 1e-8)
        }, index=resampled_data.index)
        
        # Add other features as needed (e.g., time of day, day of week)
        features_df["hour_sin"] = np.sin(2 * np.pi * features_df.index.hour / 24)
        features_df["hour_cos"] = np.cos(2 * np.pi * features_df.index.hour / 24)
        features_df["day_of_week"] = features_df.index.dayofweek / 6
        
        # Handle NaNs introduced by rolling windows
        features_df = features_df.fillna(0)
        
        self.logger.info(f"Computed features shape: {features_df.shape}")
        
        return features_df
    
    def prepare_training_data(self, features_df: pd.DataFrame, seq_length: int = 100, 
                             train_split: float = 0.7, val_split: float = 0.15) -> Tuple:
        """
        Prepare training, validation, and test datasets.
        
        Args:
            features_df: DataFrame of computed features
            seq_length: Length of sequences for the autoencoder
            train_split: Proportion of data for training
            val_split: Proportion of data for validation
            
        Returns:
            Tuple: (train_data, val_data, test_data, scaler)
        """
        from sklearn.preprocessing import StandardScaler
        
        self.logger.info(f"Preparing training data with seq_length={seq_length}")
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_df)
        
        # Create sequences
        num_samples = len(scaled_features) - seq_length + 1
        num_features = scaled_features.shape[1]
        
        sequences = np.zeros((num_samples, seq_length, num_features))
        for i in range(num_samples):
            sequences[i] = scaled_features[i : i + seq_length]
            
        # Split data
        n_train = int(num_samples * train_split)
        n_val = int(num_samples * val_split)
        
        train_data = sequences[:n_train]
        val_data = sequences[n_train : n_train + n_val]
        test_data = sequences[n_train + n_val :]
        
        self.logger.info(f"Prepared data shapes: train={train_data.shape}, val={val_data.shape}, test={test_data.shape}")
        
        return train_data, val_data, test_data, scaler


# Example usage
if __name__ == "__main__":
    import yaml
    
    # Load configuration
    with open("/home/ubuntu/adaptive_market_making_implementation/config/data_sources.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize and execute pipeline
    pipeline = FeatureEngineeringPipeline(config)
    
    try:
        pipeline.build_feature_engineering_pipeline()
        pipeline.execute()
    except KeyboardInterrupt:
        print("Pipeline execution interrupted")
    except Exception as e:
        print(f"Error executing pipeline: {str(e)}")
