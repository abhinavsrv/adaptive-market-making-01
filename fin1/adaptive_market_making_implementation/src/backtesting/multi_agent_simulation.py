"""
Multi-agent simulation for market microstructure analysis.
This module implements a multi-agent simulation framework for testing market making strategies.
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
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketAgent(ABC):
    """
    Abstract base class for market agents.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """
        Initialize the market agent.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Dictionary containing configuration parameters
        """
        self.agent_id = agent_id
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}.{agent_id}")
        
        # State variables
        self.cash = config.get("initial_cash", 1000000.0)
        self.inventory = config.get("initial_inventory", 0)
        self.trades = []
        self.orders = []
        
        self.logger.info(f"Initialized agent {agent_id}")
    
    @abstractmethod
    def on_market_update(self, market_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process market update and generate orders.
        
        Args:
            market_state: Current market state
            
        Returns:
            List[Dict[str, Any]]: List of orders to submit
        """
        pass
    
    def on_trade_execution(self, trade: Dict[str, Any]) -> None:
        """
        Process trade execution.
        
        Args:
            trade: Dictionary with trade details
        """
        # Extract trade details
        price = trade["price"]
        size = trade["size"]
        side = trade["side"]  # "buy" or "sell" from this agent's perspective
        timestamp = trade.get("timestamp", time.time())
        
        # Update inventory and cash
        if side == "buy":
            self.inventory += size
            self.cash -= price * size
        elif side == "sell":
            self.inventory -= size
            self.cash += price * size
        
        # Record trade
        self.trades.append({
            "timestamp": timestamp,
            "price": price,
            "size": size,
            "side": side,
            "inventory": self.inventory,
            "cash": self.cash
        })
        
        self.logger.debug(f"Executed trade: {side} {size} @ {price}, inventory={self.inventory}, cash={self.cash:.2f}")
    
    def get_pnl(self, mark_price: float) -> float:
        """
        Calculate current PnL.
        
        Args:
            mark_price: Current mark price for inventory valuation
            
        Returns:
            float: Current PnL
        """
        # Calculate mark-to-market value of inventory
        inventory_value = self.inventory * mark_price
        
        # Calculate PnL as cash + inventory value
        pnl = self.cash + inventory_value
        
        return pnl


class MarketMakerAgent(MarketAgent):
    """
    Market maker agent using the adaptive market making strategy.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """
        Initialize the market maker agent.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Dictionary containing configuration parameters
        """
        super().__init__(agent_id, config)
        
        # Initialize market maker strategy
        from src.strategy.adaptive_market_maker import AdaptiveMarketMaker
        self.strategy = AdaptiveMarketMaker(config)
        
        # Trading parameters
        self.tick_size = config["trading"]["tick_size"]
        self.order_timeout = config.get("order_timeout", 10)  # Order timeout in seconds
        
        # Active orders
        self.active_orders = {}
    
    def on_market_update(self, market_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process market update and generate orders.
        
        Args:
            market_state: Current market state
            
        Returns:
            List[Dict[str, Any]]: List of orders to submit
        """
        # Update strategy with market data
        quotes = self.strategy.on_market_data(market_state)
        
        # Cancel existing orders
        orders = [{"type": "cancel", "order_id": order_id} for order_id in self.active_orders.keys()]
        self.active_orders = {}
        
        # Create new orders if quotes were generated
        if quotes:
            timestamp = market_state.get("timestamp", time.time())
            
            # Create bid order
            bid_order = {
                "type": "limit",
                "side": "buy",
                "price": quotes["bid_price"],
                "size": quotes["bid_size"],
                "agent_id": self.agent_id,
                "timestamp": timestamp,
                "expiry": timestamp + self.order_timeout
            }
            bid_order_id = f"{self.agent_id}_bid_{timestamp}"
            self.active_orders[bid_order_id] = bid_order
            orders.append({"type": "submit", "order": bid_order, "order_id": bid_order_id})
            
            # Create ask order
            ask_order = {
                "type": "limit",
                "side": "sell",
                "price": quotes["ask_price"],
                "size": quotes["ask_size"],
                "agent_id": self.agent_id,
                "timestamp": timestamp,
                "expiry": timestamp + self.order_timeout
            }
            ask_order_id = f"{self.agent_id}_ask_{timestamp}"
            self.active_orders[ask_order_id] = ask_order
            orders.append({"type": "submit", "order": ask_order, "order_id": ask_order_id})
            
            # Record orders
            self.orders.append({
                "timestamp": timestamp,
                "bid_price": quotes["bid_price"],
                "ask_price": quotes["ask_price"],
                "bid_size": quotes["bid_size"],
                "ask_size": quotes["ask_size"],
                "regime": quotes.get("regime", "unknown"),
                "inventory": self.inventory
            })
        
        return orders


class InformedTraderAgent(MarketAgent):
    """
    Informed trader agent with private information about future price movements.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """
        Initialize the informed trader agent.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Dictionary containing configuration parameters
        """
        super().__init__(agent_id, config)
        
        # Trading parameters
        self.information_horizon = config.get("information_horizon", 100)  # How far ahead the agent can see
        self.information_quality = config.get("information_quality", 0.7)  # How accurate the information is
        self.trade_frequency = config.get("trade_frequency", 0.2)  # Probability of trading on each update
        self.max_trade_size = config.get("max_trade_size", 10)
        
        # State variables
        self.future_price_belief = None
        self.last_trade_time = 0
        self.min_trade_interval = config.get("min_trade_interval", 30)  # Minimum seconds between trades
    
    def on_market_update(self, market_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process market update and generate orders.
        
        Args:
            market_state: Current market state
            
        Returns:
            List[Dict[str, Any]]: List of orders to submit
        """
        orders = []
        timestamp = market_state.get("timestamp", time.time())
        
        # Check if enough time has passed since last trade
        if timestamp - self.last_trade_time < self.min_trade_interval:
            return orders
        
        # Update future price belief
        self._update_price_belief(market_state)
        
        # Decide whether to trade
        if np.random.random() < self.trade_frequency:
            # Get current market prices
            bid_price = market_state.get("bid_price")
            ask_price = market_state.get("ask_price")
            mid_price = market_state.get("mid_price")
            
            if bid_price is None or ask_price is None:
                if mid_price is not None:
                    # Estimate bid/ask from mid price
                    spread = mid_price * 0.0005  # Estimated half-spread as 0.05% of price
                    bid_price = mid_price - spread
                    ask_price = mid_price + spread
                else:
                    # No price information available
                    return orders
            
            # Determine trade direction based on future price belief
            if self.future_price_belief > ask_price * 1.0005:  # Expected profit threshold
                # Buy if future price is higher
                side = "buy"
                price = ask_price
                expected_profit = self.future_price_belief - ask_price
            elif self.future_price_belief < bid_price * 0.9995:  # Expected profit threshold
                # Sell if future price is lower
                side = "sell"
                price = bid_price
                expected_profit = bid_price - self.future_price_belief
            else:
                # No profitable opportunity
                return orders
            
            # Determine trade size based on expected profit
            # More aggressive for larger expected profits
            max_size = min(self.max_trade_size, 20)  # Cap at 20 contracts
            size_factor = min(expected_profit / price * 1000, 1.0)  # Scale by expected profit
            size = max(1, int(max_size * size_factor))
            
            # Create market order
            order = {
                "type": "market",
                "side": side,
                "size": size,
                "agent_id": self.agent_id,
                "timestamp": timestamp
            }
            order_id = f"{self.agent_id}_{side}_{timestamp}"
            orders.append({"type": "submit", "order": order, "order_id": order_id})
            
            # Update last trade time
            self.last_trade_time = timestamp
            
            self.logger.debug(f"Submitting {side} order for {size} contracts, expected profit: {expected_profit:.4f}")
        
        return orders
    
    def _update_price_belief(self, market_state: Dict[str, Any]) -> None:
        """
        Update belief about future price.
        
        Args:
            market_state: Current market state
        """
        # Get current price
        mid_price = market_state.get("mid_price")
        if mid_price is None:
            bid_price = market_state.get("bid_price")
            ask_price = market_state.get("ask_price")
            if bid_price is not None and ask_price is not None:
                mid_price = (bid_price + ask_price) / 2
            else:
                mid_price = market_state.get("last_price")
        
        if mid_price is None:
            # No price information available
            return
        
        # Get future price from simulation if available
        future_price = market_state.get("future_price")
        
        if future_price is not None:
            # Blend with noise based on information quality
            noise = np.random.normal(0, mid_price * 0.01 * (1 - self.information_quality))
            self.future_price_belief = future_price * self.information_quality + (mid_price + noise) * (1 - self.information_quality)
        else:
            # Generate synthetic future price belief
            drift = np.random.normal(0, 0.001)  # Small random drift
            volatility = market_state.get("volatility", 0.15)  # Annualized volatility
            
            # Convert to per-second volatility
            seconds_per_year = 252 * 6.5 * 3600
            vol_per_second = volatility / np.sqrt(seconds_per_year)
            
            # Project forward
            horizon_seconds = self.information_horizon
            projected_change = drift * horizon_seconds + np.random.normal(0, vol_per_second * np.sqrt(horizon_seconds))
            self.future_price_belief = mid_price * (1 + projected_change)


class NoiseTraderAgent(MarketAgent):
    """
    Noise trader agent that trades randomly.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """
        Initialize the noise trader agent.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Dictionary containing configuration parameters
        """
        super().__init__(agent_id, config)
        
        # Trading parameters
        self.trade_frequency = config.get("trade_frequency", 0.1)  # Probability of trading on each update
        self.max_trade_size = config.get("max_trade_size", 5)
        self.limit_order_ratio = config.get("limit_order_ratio", 0.7)  # Ratio of limit to market orders
        self.order_timeout = config.get("order_timeout", 60)  # Order timeout in seconds
        
        # State variables
        self.last_trade_time = 0
        self.min_trade_interval = config.get("min_trade_interval", 10)  # Minimum seconds between trades
    
    def on_market_update(self, market_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process market update and generate orders.
        
        Args:
            market_state: Current market state
            
        Returns:
            List[Dict[str, Any]]: List of orders to submit
        """
        orders = []
        timestamp = market_state.get("timestamp", time.time())
        
        # Check if enough time has passed since last trade
        if timestamp - self.last_trade_time < self.min_trade_interval:
            return orders
        
        # Decide whether to trade
        if np.random.random() < self.trade_frequency:
            # Get current market prices
            bid_price = market_state.get("bid_price")
            ask_price = market_state.get("ask_price")
            mid_price = market_state.get("mid_price")
            
            if bid_price is None or ask_price is None:
                if mid_price is not None:
                    # Estimate bid/ask from mid price
                    spread = mid_price * 0.0005  # Estimated half-spread as 0.05% of price
                    bid_price = mid_price - spread
                    ask_price = mid_price + spread
                else:
                    # No price information available
                    return orders
            
            # Determine trade side randomly
            side = "buy" if np.random.random() < 0.5 else "sell"
            
            # Determine trade size randomly
            size = np.random.randint(1, self.max_trade_size + 1)
            
            # Determine order type
            if np.random.random() < self.limit_order_ratio:
                # Limit order
                if side == "buy":
                    # Place bid below current bid
                    price = bid_price * (1 - np.random.uniform(0.0001, 0.001))
                else:
                    # Place ask above current ask
                    price = ask_price * (1 + np.random.uniform(0.0001, 0.001))
                
                # Round to tick size
                tick_size = market_state.get("tick_size", 0.01)
                price = round(price / tick_size) * tick_size
                
                # Create limit order
                order = {
                    "type": "limit",
                    "side": side,
                    "price": price,
                    "size": size,
                    "agent_id": self.agent_id,
                    "timestamp": timestamp,
                    "expiry": timestamp + self.order_timeout
                }
            else:
                # Market order
                order = {
                    "type": "market",
                    "side": side,
                    "size": size,
                    "agent_id": self.agent_id,
                    "timestamp": timestamp
                }
            
            order_id = f"{self.agent_id}_{side}_{timestamp}"
            orders.append({"type": "submit", "order": order, "order_id": order_id})
            
            # Update last trade time
            self.last_trade_time = timestamp
            
            self.logger.debug(f"Submitting {order['type']} {side} order for {size} contracts")
        
        return orders


class MarketSimulator:
    """
    Market simulator for multi-agent simulations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the market simulator.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__ + '.MarketSimulator')
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Initialize order book
        self.order_book = {
            "bids": {},  # price -> [orders]
            "asks": {}   # price -> [orders]
        }
        
        # Initialize market state
        self.market_state = {
            "timestamp": time.time(),
            "mid_price": config["simulation"]["initial_price"],
            "bid_price": config["simulation"]["initial_price"] * 0.999,
            "ask_price": config["simulation"]["initial_price"] * 1.001,
            "last_price": config["simulation"]["initial_price"],
            "volatility": config["simulation"]["initial_volatility"],
            "tick_size": config["trading"]["tick_size"],
            "features": np.zeros(config["model_params"]["autoencoder"]["input_channels"])
        }
        
        # Simulation parameters
        self.price_path = self._generate_price_path()
        self.current_step = 0
        self.trades = []
        
        self.logger.info("Initialized market simulator")
    
    def _initialize_agents(self) -> Dict[str, MarketAgent]:
        """
        Initialize market agents.
        
        Returns:
            Dict[str, MarketAgent]: Dictionary of initialized agents
        """
        agents = {}
        
        # Initialize market maker
        mm_config = self.config.copy()
        mm_config["initial_cash"] = self.config["simulation"]["mm_initial_cash"]
        mm_config["initial_inventory"] = self.config["simulation"]["mm_initial_inventory"]
        agents["market_maker"] = MarketMakerAgent("market_maker", mm_config)
        
        # Initialize informed traders
        num_informed = self.config["simulation"]["num_informed_traders"]
        for i in range(num_informed):
            agent_id = f"informed_{i}"
            agent_config = self.config.copy()
            agent_config["initial_cash"] = self.config["simulation"]["trader_initial_cash"]
            agent_config["information_quality"] = 0.5 + np.random.random() * 0.4  # Random quality between 0.5 and 0.9
            agents[agent_id] = InformedTraderAgent(agent_id, agent_config)
        
        # Initialize noise traders
        num_noise = self.config["simulation"]["num_noise_traders"]
        for i in range(num_noise):
            agent_id = f"noise_{i}"
            agent_config = self.config.copy()
            agent_config["initial_cash"] = self.config["simulation"]["trader_initial_cash"]
            agent_config["trade_frequency"] = 0.05 + np.random.random() * 0.15  # Random frequency between 0.05 and 0.2
            agents[agent_id] = NoiseTraderAgent(agent_id, agent_config)
        
        self.logger.info(f"Initialized {len(agents)} agents: 1 market maker, {num_informed} informed traders, {num_noise} noise traders")
        
        return agents
    
    def _generate_price_path(self) -> np.ndarray:
        """
        Generate price path for simulation.
        
        Returns:
            np.ndarray: Generated price path
        """
        # Simulation parameters
        num_steps = self.config["simulation"]["num_steps"]
        initial_price = self.config["simulation"]["initial_price"]
        volatility = self.config["simulation"]["initial_volatility"]
        
        # Convert annual volatility to per-step volatility
        steps_per_year = 252 * 6.5 * 3600  # Assuming 1-second steps
        vol_per_step = volatility / np.sqrt(steps_per_year)
        
        # Generate random walk
        returns = np.random.normal(0, vol_per_step, num_steps)
        
        # Add occasional jumps
        jump_prob = 0.001  # Probability of jump per step
        jump_size_mean = 0.0
        jump_size_std = 0.005
        
        jumps = np.random.binomial(1, jump_prob, num_steps)
        jump_sizes = np.random.normal(jump_size_mean, jump_size_std, num_steps)
        returns += jumps * jump_sizes
        
        # Convert returns to prices
        price_path = initial_price * np.cumprod(1 + returns)
        
        self.logger.info(f"Generated price path with {num_steps} steps, initial={initial_price:.2f}, final={price_path[-1]:.2f}")
        
        return price_path
    
    def _update_market_state(self) -> None:
        """
        Update market state for current step.
        """
        # Get current price from price path
        current_price = self.price_path[self.current_step]
        
        # Update market state
        self.market_state["timestamp"] = time.time()
        self.market_state["mid_price"] = current_price
        
        # Update bid/ask prices based on order book
        best_bid = self._get_best_bid()
        best_ask = self._get_best_ask()
        
        if best_bid is not None:
            self.market_state["bid_price"] = best_bid
        else:
            self.market_state["bid_price"] = current_price * 0.999
        
        if best_ask is not None:
            self.market_state["ask_price"] = best_ask
        else:
            self.market_state["ask_price"] = current_price * 1.001
        
        # Calculate volatility (simple rolling standard deviation of returns)
        window_size = 20
        if self.current_step >= window_size:
            recent_prices = self.price_path[self.current_step - window_size:self.current_step]
            returns = np.diff(np.log(recent_prices))
            vol = np.std(returns) * np.sqrt(252 * 6.5 * 3600)  # Annualized
            self.market_state["volatility"] = vol
        
        # Add future price for informed traders
        look_ahead = min(100, len(self.price_path) - self.current_step - 1)
        if look_ahead > 0:
            self.market_state["future_price"] = self.price_path[self.current_step + look_ahead]
        
        # Generate features
        self.market_state["features"] = self._generate_features()
    
    def _generate_features(self) -> np.ndarray:
        """
        Generate feature vector for current market state.
        
        Returns:
            np.ndarray: Feature vector
        """
        # Get feature dimension from config
        input_channels = self.config["model_params"]["autoencoder"]["input_channels"]
        
        # Create feature vector
        features = np.zeros(input_channels)
        
        # Basic features
        features[0] = self.market_state["mid_price"] / 1000  # Normalized price
        features[1] = self.market_state["volatility"] * 10  # Scaled volatility
        
        # Calculate return
        if self.current_step > 0:
            prev_price = self.price_path[self.current_step - 1]
            current_price = self.price_path[self.current_step]
            ret = np.log(current_price / prev_price)
            features[2] = ret * 100  # Scaled return
        
        # Add order book imbalance
        bid_volume = sum(order["size"] for orders in self.order_book["bids"].values() for order in orders)
        ask_volume = sum(order["size"] for orders in self.order_book["asks"].values() for order in orders)
        total_volume = bid_volume + ask_volume
        
        if total_volume > 0:
            imbalance = (bid_volume - ask_volume) / total_volume
            features[3] = imbalance
        
        # Add some autocorrelated noise for the rest
        if hasattr(self, "prev_features"):
            prev_features = self.prev_features[4:]
            noise = prev_features * 0.8 + np.random.randn(input_channels - 4) * 0.2
        else:
            noise = np.random.randn(input_channels - 4)
        
        features[4:] = noise
        
        # Store for next iteration
        self.prev_features = features.copy()
        
        return features
    
    def _get_best_bid(self) -> Optional[float]:
        """
        Get best bid price from order book.
        
        Returns:
            Optional[float]: Best bid price or None if no bids
        """
        if not self.order_book["bids"]:
            return None
        
        return max(self.order_book["bids"].keys())
    
    def _get_best_ask(self) -> Optional[float]:
        """
        Get best ask price from order book.
        
        Returns:
            Optional[float]: Best ask price or None if no asks
        """
        if not self.order_book["asks"]:
            return None
        
        return min(self.order_book["asks"].keys())
    
    def _process_order(self, order: Dict[str, Any], order_id: str) -> List[Dict[str, Any]]:
        """
        Process an order and generate trades.
        
        Args:
            order: Order to process
            order_id: Order ID
            
        Returns:
            List[Dict[str, Any]]: List of generated trades
        """
        trades = []
        
        # Process based on order type
        if order["type"] == "market":
            # Market order
            trades = self._process_market_order(order, order_id)
        elif order["type"] == "limit":
            # Limit order
            trades = self._process_limit_order(order, order_id)
        
        return trades
    
    def _process_market_order(self, order: Dict[str, Any], order_id: str) -> List[Dict[str, Any]]:
        """
        Process a market order.
        
        Args:
            order: Market order to process
            order_id: Order ID
            
        Returns:
            List[Dict[str, Any]]: List of generated trades
        """
        trades = []
        remaining_size = order["size"]
        
        if order["side"] == "buy":
            # Buy order matches against asks
            sorted_prices = sorted(self.order_book["asks"].keys())
            
            for price in sorted_prices:
                if remaining_size <= 0:
                    break
                
                # Match against orders at this price level
                orders_at_price = self.order_book["asks"][price]
                
                # Process each order
                i = 0
                while i < len(orders_at_price) and remaining_size > 0:
                    matched_order = orders_at_price[i]
                    
                    # Calculate trade size
                    trade_size = min(remaining_size, matched_order["size"])
                    
                    # Create trade
                    trade = {
                        "timestamp": order["timestamp"],
                        "price": price,
                        "size": trade_size,
                        "aggressor_side": "buy",
                        "aggressor_id": order["agent_id"],
                        "passive_id": matched_order["agent_id"]
                    }
                    trades.append(trade)
                    
                    # Update remaining size
                    remaining_size -= trade_size
                    
                    # Update matched order size
                    matched_order["size"] -= trade_size
                    
                    # Remove matched order if fully filled
                    if matched_order["size"] <= 0:
                        orders_at_price.pop(i)
                    else:
                        i += 1
                
                # Remove price level if empty
                if not orders_at_price:
                    del self.order_book["asks"][price]
        
        elif order["side"] == "sell":
            # Sell order matches against bids
            sorted_prices = sorted(self.order_book["bids"].keys(), reverse=True)
            
            for price in sorted_prices:
                if remaining_size <= 0:
                    break
                
                # Match against orders at this price level
                orders_at_price = self.order_book["bids"][price]
                
                # Process each order
                i = 0
                while i < len(orders_at_price) and remaining_size > 0:
                    matched_order = orders_at_price[i]
                    
                    # Calculate trade size
                    trade_size = min(remaining_size, matched_order["size"])
                    
                    # Create trade
                    trade = {
                        "timestamp": order["timestamp"],
                        "price": price,
                        "size": trade_size,
                        "aggressor_side": "sell",
                        "aggressor_id": order["agent_id"],
                        "passive_id": matched_order["agent_id"]
                    }
                    trades.append(trade)
                    
                    # Update remaining size
                    remaining_size -= trade_size
                    
                    # Update matched order size
                    matched_order["size"] -= trade_size
                    
                    # Remove matched order if fully filled
                    if matched_order["size"] <= 0:
                        orders_at_price.pop(i)
                    else:
                        i += 1
                
                # Remove price level if empty
                if not orders_at_price:
                    del self.order_book["bids"][price]
        
        # If order not fully filled, execute at market price
        if remaining_size > 0:
            # Use mid price as fallback
            price = self.market_state["mid_price"]
            
            # Create trade
            trade = {
                "timestamp": order["timestamp"],
                "price": price,
                "size": remaining_size,
                "aggressor_side": order["side"],
                "aggressor_id": order["agent_id"],
                "passive_id": "market"  # Market provides liquidity
            }
            trades.append(trade)
        
        return trades
    
    def _process_limit_order(self, order: Dict[str, Any], order_id: str) -> List[Dict[str, Any]]:
        """
        Process a limit order.
        
        Args:
            order: Limit order to process
            order_id: Order ID
            
        Returns:
            List[Dict[str, Any]]: List of generated trades
        """
        trades = []
        
        if order["side"] == "buy":
            # Check if order crosses the spread
            best_ask = self._get_best_ask()
            
            if best_ask is not None and order["price"] >= best_ask:
                # Convert to market order for crossing portion
                market_order = order.copy()
                market_order["type"] = "market"
                trades = self._process_market_order(market_order, order_id)
                
                # Calculate remaining size
                filled_size = sum(trade["size"] for trade in trades)
                remaining_size = order["size"] - filled_size
                
                # Add remaining size to order book
                if remaining_size > 0:
                    order_copy = order.copy()
                    order_copy["size"] = remaining_size
                    
                    if order["price"] not in self.order_book["bids"]:
                        self.order_book["bids"][order["price"]] = []
                    
                    self.order_book["bids"][order["price"]].append(order_copy)
            else:
                # Add to order book
                if order["price"] not in self.order_book["bids"]:
                    self.order_book["bids"][order["price"]] = []
                
                self.order_book["bids"][order["price"]].append(order)
        
        elif order["side"] == "sell":
            # Check if order crosses the spread
            best_bid = self._get_best_bid()
            
            if best_bid is not None and order["price"] <= best_bid:
                # Convert to market order for crossing portion
                market_order = order.copy()
                market_order["type"] = "market"
                trades = self._process_market_order(market_order, order_id)
                
                # Calculate remaining size
                filled_size = sum(trade["size"] for trade in trades)
                remaining_size = order["size"] - filled_size
                
                # Add remaining size to order book
                if remaining_size > 0:
                    order_copy = order.copy()
                    order_copy["size"] = remaining_size
                    
                    if order["price"] not in self.order_book["asks"]:
                        self.order_book["asks"][order["price"]] = []
                    
                    self.order_book["asks"][order["price"]].append(order_copy)
            else:
                # Add to order book
                if order["price"] not in self.order_book["asks"]:
                    self.order_book["asks"][order["price"]] = []
                
                self.order_book["asks"][order["price"]].append(order)
        
        return trades
    
    def _clean_expired_orders(self) -> None:
        """
        Remove expired orders from the order book.
        """
        current_time = time.time()
        
        # Clean bids
        for price in list(self.order_book["bids"].keys()):
            orders = self.order_book["bids"][price]
            self.order_book["bids"][price] = [order for order in orders if "expiry" not in order or order["expiry"] > current_time]
            
            if not self.order_book["bids"][price]:
                del self.order_book["bids"][price]
        
        # Clean asks
        for price in list(self.order_book["asks"].keys()):
            orders = self.order_book["asks"][price]
            self.order_book["asks"][price] = [order for order in orders if "expiry" not in order or order["expiry"] > current_time]
            
            if not self.order_book["asks"][price]:
                del self.order_book["asks"][price]
    
    def run_simulation(self) -> Dict[str, Any]:
        """
        Run the market simulation.
        
        Returns:
            Dict[str, Any]: Simulation results
        """
        try:
            self.logger.info("Starting market simulation")
            
            # Track execution time
            start_time = time.time()
            
            # Run for specified number of steps
            num_steps = self.config["simulation"]["num_steps"]
            
            for step in range(num_steps):
                self.current_step = step
                
                # Update market state
                self._update_market_state()
                
                # Clean expired orders
                self._clean_expired_orders()
                
                # Process agent actions
                all_trades = []
                
                for agent_id, agent in self.agents.items():
                    # Get agent orders
                    orders = agent.on_market_update(self.market_state.copy())
                    
                    # Process orders
                    for order_action in orders:
                        if order_action["type"] == "submit":
                            # Submit new order
                            order = order_action["order"]
                            order_id = order_action["order_id"]
                            
                            # Process order
                            trades = self._process_order(order, order_id)
                            all_trades.extend(trades)
                        
                        elif order_action["type"] == "cancel":
                            # Cancel existing order
                            # In a real implementation, this would remove the order from the book
                            # For this prototype, we'll just log it
                            self.logger.debug(f"Cancelling order {order_action['order_id']}")
                
                # Process trades
                for trade in all_trades:
                    # Update market state with last trade price
                    self.market_state["last_price"] = trade["price"]
                    
                    # Notify agents involved in the trade
                    aggressor_id = trade["aggressor_id"]
                    passive_id = trade["passive_id"]
                    
                    if aggressor_id in self.agents:
                        # Aggressor executed a trade
                        agent_trade = {
                            "timestamp": trade["timestamp"],
                            "price": trade["price"],
                            "size": trade["size"],
                            "side": trade["aggressor_side"]
                        }
                        self.agents[aggressor_id].on_trade_execution(agent_trade)
                    
                    if passive_id in self.agents:
                        # Passive side executed a trade
                        passive_side = "sell" if trade["aggressor_side"] == "buy" else "buy"
                        agent_trade = {
                            "timestamp": trade["timestamp"],
                            "price": trade["price"],
                            "size": trade["size"],
                            "side": passive_side
                        }
                        self.agents[passive_id].on_trade_execution(agent_trade)
                
                # Record trades
                self.trades.extend(all_trades)
                
                # Log progress periodically
                if step % 100 == 0 or step == num_steps - 1:
                    elapsed = time.time() - start_time
                    self.logger.info(f"Completed step {step+1}/{num_steps} ({(step+1)/num_steps*100:.1f}%) in {elapsed:.1f}s")
            
            # Calculate final metrics
            results = self._calculate_metrics()
            
            self.logger.info(f"Simulation completed in {time.time() - start_time:.1f}s with {len(self.trades)} trades")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in simulation: {str(e)}")
            raise
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate simulation metrics.
        
        Returns:
            Dict[str, Any]: Simulation metrics
        """
        metrics = {
            "simulation_steps": self.current_step + 1,
            "total_trades": len(self.trades),
            "final_price": self.market_state["mid_price"],
            "agent_metrics": {}
        }
        
        # Calculate agent-specific metrics
        for agent_id, agent in self.agents.items():
            agent_metrics = {
                "final_inventory": agent.inventory,
                "final_cash": agent.cash,
                "pnl": agent.get_pnl(self.market_state["mid_price"]),
                "num_trades": len(agent.trades)
            }
            
            metrics["agent_metrics"][agent_id] = agent_metrics
        
        return metrics
    
    def save_results(self, base_path: str) -> None:
        """
        Save simulation results to files.
        
        Args:
            base_path: Base path for saving files
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(base_path, exist_ok=True)
            
            # Save price path
            price_df = pd.DataFrame({
                "step": range(len(self.price_path)),
                "price": self.price_path
            })
            price_df.to_csv(f"{base_path}/price_path.csv", index=False)
            
            # Save trades
            if self.trades:
                trades_df = pd.DataFrame(self.trades)
                trades_df.to_csv(f"{base_path}/trades.csv", index=False)
            
            # Save agent results
            for agent_id, agent in self.agents.items():
                agent_dir = f"{base_path}/{agent_id}"
                os.makedirs(agent_dir, exist_ok=True)
                
                # Save trades
                if agent.trades:
                    agent_trades_df = pd.DataFrame(agent.trades)
                    agent_trades_df.to_csv(f"{agent_dir}/trades.csv", index=False)
                
                # Save orders for market maker
                if isinstance(agent, MarketMakerAgent) and agent.orders:
                    orders_df = pd.DataFrame(agent.orders)
                    orders_df.to_csv(f"{agent_dir}/orders.csv", index=False)
            
            # Save metrics
            metrics = self._calculate_metrics()
            with open(f"{base_path}/metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            
            self.logger.info(f"Simulation results saved to {base_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save simulation results: {str(e)}")
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """
        Plot simulation results.
        
        Args:
            save_path: Optional path to save plots
        """
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
            
            # Plot price path
            axes[0].plot(self.price_path, 'k-')
            axes[0].set_title('Price Path')
            axes[0].set_ylabel('Price')
            axes[0].grid(True)
            
            # Plot trades
            if self.trades:
                trade_df = pd.DataFrame(self.trades)
                
                # Buy trades (from market maker perspective)
                buy_trades = trade_df[
                    (trade_df['aggressor_id'] == 'market_maker') & 
                    (trade_df['aggressor_side'] == 'buy')
                ]
                if not buy_trades.empty:
                    axes[0].scatter(
                        buy_trades.index, 
                        buy_trades['price'], 
                        color='g', 
                        marker='^', 
                        alpha=0.7, 
                        label='MM Buy'
                    )
                
                # Sell trades (from market maker perspective)
                sell_trades = trade_df[
                    (trade_df['aggressor_id'] == 'market_maker') & 
                    (trade_df['aggressor_side'] == 'sell')
                ]
                if not sell_trades.empty:
                    axes[0].scatter(
                        sell_trades.index, 
                        sell_trades['price'], 
                        color='r', 
                        marker='v', 
                        alpha=0.7, 
                        label='MM Sell'
                    )
                
                # Passive trades (from market maker perspective)
                passive_trades = trade_df[trade_df['passive_id'] == 'market_maker']
                if not passive_trades.empty:
                    axes[0].scatter(
                        passive_trades.index, 
                        passive_trades['price'], 
                        color='b', 
                        marker='o', 
                        alpha=0.5, 
                        label='MM Passive'
                    )
                
                axes[0].legend()
            
            # Plot market maker inventory
            if 'market_maker' in self.agents:
                mm = self.agents['market_maker']
                if mm.trades:
                    mm_trades_df = pd.DataFrame(mm.trades)
                    axes[1].plot(mm_trades_df.index, mm_trades_df['inventory'], 'b-')
                    axes[1].set_title('Market Maker Inventory')
                    axes[1].set_ylabel('Position')
                    axes[1].grid(True)
            
            # Plot market maker PnL
            if 'market_maker' in self.agents:
                mm = self.agents['market_maker']
                if mm.trades:
                    mm_trades_df = pd.DataFrame(mm.trades)
                    
                    # Calculate PnL at each trade
                    pnl = []
                    for i, row in mm_trades_df.iterrows():
                        mark_price = self.price_path[min(i, len(self.price_path) - 1)]
                        pnl.append(row['cash'] + row['inventory'] * mark_price)
                    
                    axes[2].plot(mm_trades_df.index, pnl, 'g-')
                    axes[2].set_title('Market Maker PnL')
                    axes[2].set_ylabel('PnL')
                    axes[2].grid(True)
            
            # Format x-axis
            axes[2].set_xlabel('Simulation Step')
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
        "simulation": {
            "num_steps": 1000,
            "initial_price": 4500.0,
            "initial_volatility": 0.15,
            "num_informed_traders": 2,
            "num_noise_traders": 5,
            "mm_initial_cash": 1000000.0,
            "mm_initial_inventory": 0,
            "trader_initial_cash": 500000.0
        }
    }
    
    # Initialize simulator
    simulator = MarketSimulator(config)
    
    # Run simulation
    results = simulator.run_simulation()
    
    # Print results
    print(json.dumps(results, indent=2))
    
    # Save results
    simulator.save_results("/home/ubuntu/adaptive_market_making_implementation/results/simulation")
    
    # Plot results
    simulator.plot_results("/home/ubuntu/adaptive_market_making_implementation/results/simulation/simulation_plot.png")
