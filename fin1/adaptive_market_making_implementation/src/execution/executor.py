"""
Execution module for the adaptive market making strategy.
This module handles order execution and risk management.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import time
import os
import json
import threading
import requests
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExecutionEngine:
    """
    Execution engine for the adaptive market making strategy.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the execution engine.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__ + '.ExecutionEngine')
        
        # Initialize exchange client
        self.exchange_client = self._initialize_exchange_client()
        
        # Initialize risk manager
        self.risk_manager = self._initialize_risk_manager()
        
        # State variables
        self.is_running = False
        self.execution_thread = None
        self.order_queue = []
        self.order_queue_lock = threading.Lock()
        
        # Trading parameters
        self.max_position = config.get("max_position", 100)
        self.target_inventory = config.get("target_inventory", 0)
        self.tick_size = config.get("tick_size", 0.01)
        self.contract_multiplier = config.get("contract_multiplier", 1)
        
        # Performance tracking
        self.trades = []
        self.orders = []
        self.positions = {}
        self.cash = config.get("initial_cash", 1000000.0)
        
        self.logger.info("Initialized execution engine")
    
    def _initialize_exchange_client(self) -> Any:
        """
        Initialize exchange client.
        
        Returns:
            Any: Initialized exchange client
        """
        try:
            self.logger.info("Initializing exchange client")
            
            # Get exchange configuration
            exchange_config = self.config.get("exchange", {})
            exchange_type = exchange_config.get("type", "cme")
            
            # Initialize client based on exchange type
            if exchange_type.lower() == "cme":
                client = CMEExchangeClient(exchange_config)
            elif exchange_type.lower() == "simulation":
                client = SimulationExchangeClient(exchange_config)
            else:
                raise ValueError(f"Unsupported exchange type: {exchange_type}")
            
            self.logger.info(f"Exchange client initialized for {exchange_type}")
            return client
            
        except Exception as e:
            self.logger.error(f"Failed to initialize exchange client: {str(e)}")
            raise
    
    def _initialize_risk_manager(self) -> Any:
        """
        Initialize risk manager.
        
        Returns:
            Any: Initialized risk manager
        """
        try:
            self.logger.info("Initializing risk manager")
            
            # Get risk configuration
            risk_config = self.config.get("risk", {})
            
            # Initialize risk manager
            risk_manager = RiskManager(risk_config)
            
            self.logger.info("Risk manager initialized")
            return risk_manager
            
        except Exception as e:
            self.logger.error(f"Failed to initialize risk manager: {str(e)}")
            raise
    
    def start(self) -> None:
        """
        Start the execution engine.
        """
        try:
            if self.is_running:
                self.logger.warning("Execution engine is already running")
                return
            
            self.logger.info("Starting execution engine")
            
            # Set running flag
            self.is_running = True
            
            # Create execution thread
            self.execution_thread = threading.Thread(
                target=self._execution_loop,
                daemon=True
            )
            
            # Start thread
            self.execution_thread.start()
            
            self.logger.info("Execution engine started")
            
        except Exception as e:
            self.logger.error(f"Failed to start execution engine: {str(e)}")
            self.is_running = False
            raise
    
    def stop(self) -> None:
        """
        Stop the execution engine.
        """
        try:
            if not self.is_running:
                self.logger.warning("Execution engine is not running")
                return
            
            self.logger.info("Stopping execution engine")
            
            # Clear running flag
            self.is_running = False
            
            # Wait for thread to finish
            if self.execution_thread:
                self.logger.info("Waiting for execution to stop")
                self.execution_thread.join(timeout=5.0)
                self.execution_thread = None
            
            # Cancel all open orders
            self._cancel_all_orders()
            
            self.logger.info("Execution engine stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop execution engine: {str(e)}")
            raise
    
    def submit_order(self, order: Dict[str, Any]) -> str:
        """
        Submit an order for execution.
        
        Args:
            order: Order to submit
            
        Returns:
            str: Order ID
        """
        try:
            # Generate order ID
            order_id = f"order_{int(time.time() * 1000)}_{len(self.orders)}"
            
            # Add order ID to order
            order["order_id"] = order_id
            
            # Add timestamp if not present
            if "timestamp" not in order:
                order["timestamp"] = time.time()
            
            # Add to order queue
            with self.order_queue_lock:
                self.order_queue.append(order)
            
            self.logger.info(f"Order {order_id} submitted: {order['side']} {order['size']} @ {order.get('price', 'market')}")
            
            return order_id
            
        except Exception as e:
            self.logger.error(f"Failed to submit order: {str(e)}")
            raise
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            bool: True if order was cancelled, False otherwise
        """
        try:
            # Check if order is in queue
            with self.order_queue_lock:
                for i, order in enumerate(self.order_queue):
                    if order["order_id"] == order_id:
                        # Remove from queue
                        self.order_queue.pop(i)
                        self.logger.info(f"Order {order_id} cancelled from queue")
                        return True
            
            # If not in queue, try to cancel on exchange
            if self.exchange_client:
                result = self.exchange_client.cancel_order(order_id)
                if result:
                    self.logger.info(f"Order {order_id} cancelled on exchange")
                return result
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            return False
    
    def _cancel_all_orders(self) -> None:
        """
        Cancel all open orders.
        """
        try:
            self.logger.info("Cancelling all orders")
            
            # Clear order queue
            with self.order_queue_lock:
                self.order_queue = []
            
            # Cancel orders on exchange
            if self.exchange_client:
                self.exchange_client.cancel_all_orders()
            
            self.logger.info("All orders cancelled")
            
        except Exception as e:
            self.logger.error(f"Failed to cancel all orders: {str(e)}")
    
    def _execution_loop(self) -> None:
        """
        Main execution loop.
        """
        try:
            self.logger.info("Execution loop started")
            
            # Execution loop
            while self.is_running:
                try:
                    # Process order queue
                    self._process_order_queue()
                    
                    # Update positions
                    self._update_positions()
                    
                    # Sleep to avoid excessive CPU usage
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"Error in execution loop: {str(e)}")
                    time.sleep(1.0)  # Sleep before retry
            
            self.logger.info("Execution loop stopped")
            
        except Exception as e:
            self.logger.error(f"Fatal error in execution loop: {str(e)}")
    
    def _process_order_queue(self) -> None:
        """
        Process orders in the queue.
        """
        # Get orders from queue
        orders_to_process = []
        with self.order_queue_lock:
            if self.order_queue:
                orders_to_process = self.order_queue.copy()
                self.order_queue = []
        
        # Process each order
        for order in orders_to_process:
            try:
                # Check risk limits
                if not self.risk_manager.check_order(order, self.positions):
                    self.logger.warning(f"Order {order['order_id']} rejected by risk manager")
                    continue
                
                # Submit to exchange
                if self.exchange_client:
                    result = self.exchange_client.submit_order(order)
                    
                    # Record order
                    self.orders.append({
                        "timestamp": order["timestamp"],
                        "order_id": order["order_id"],
                        "side": order["side"],
                        "type": order.get("type", "market"),
                        "price": order.get("price"),
                        "size": order["size"],
                        "status": "submitted" if result else "rejected"
                    })
                    
                    if not result:
                        self.logger.warning(f"Order {order['order_id']} rejected by exchange")
                
            except Exception as e:
                self.logger.error(f"Error processing order {order.get('order_id')}: {str(e)}")
    
    def _update_positions(self) -> None:
        """
        Update positions and process trades.
        """
        try:
            # Get new trades from exchange
            if self.exchange_client:
                new_trades = self.exchange_client.get_new_trades()
                
                # Process each trade
                for trade in new_trades:
                    # Extract trade details
                    symbol = trade.get("symbol", "unknown")
                    side = trade.get("side")
                    price = trade.get("price")
                    size = trade.get("size")
                    timestamp = trade.get("timestamp", time.time())
                    
                    # Update position
                    if symbol not in self.positions:
                        self.positions[symbol] = 0
                    
                    if side == "buy":
                        self.positions[symbol] += size
                        self.cash -= price * size * self.contract_multiplier
                    elif side == "sell":
                        self.positions[symbol] -= size
                        self.cash += price * size * self.contract_multiplier
                    
                    # Record trade
                    self.trades.append({
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "side": side,
                        "price": price,
                        "size": size,
                        "position": self.positions[symbol],
                        "cash": self.cash
                    })
                    
                    self.logger.info(f"Trade executed: {side} {size} {symbol} @ {price}, position={self.positions[symbol]}, cash={self.cash:.2f}")
            
        except Exception as e:
            self.logger.error(f"Failed to update positions: {str(e)}")
    
    def get_position(self, symbol: str) -> int:
        """
        Get current position for a symbol.
        
        Args:
            symbol: Symbol to get position for
            
        Returns:
            int: Current position
        """
        return self.positions.get(symbol, 0)
    
    def get_cash(self) -> float:
        """
        Get current cash balance.
        
        Returns:
            float: Current cash balance
        """
        return self.cash
    
    def get_pnl(self, mark_prices: Dict[str, float]) -> float:
        """
        Calculate current PnL.
        
        Args:
            mark_prices: Dictionary of mark prices by symbol
            
        Returns:
            float: Current PnL
        """
        # Start with cash
        pnl = self.cash
        
        # Add mark-to-market value of positions
        for symbol, position in self.positions.items():
            if symbol in mark_prices:
                mark_price = mark_prices[symbol]
                position_value = position * mark_price * self.contract_multiplier
                pnl += position_value
        
        return pnl


class ExchangeClient:
    """
    Abstract base class for exchange clients.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the exchange client.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def submit_order(self, order: Dict[str, Any]) -> bool:
        """
        Submit an order to the exchange.
        
        Args:
            order: Order to submit
            
        Returns:
            bool: True if order was submitted successfully, False otherwise
        """
        raise NotImplementedError("Subclasses must implement submit_order()")
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order on the exchange.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            bool: True if order was cancelled, False otherwise
        """
        raise NotImplementedError("Subclasses must implement cancel_order()")
    
    def cancel_all_orders(self) -> bool:
        """
        Cancel all open orders on the exchange.
        
        Returns:
            bool: True if orders were cancelled, False otherwise
        """
        raise NotImplementedError("Subclasses must implement cancel_all_orders()")
    
    def get_new_trades(self) -> List[Dict[str, Any]]:
        """
        Get new trades from the exchange.
        
        Returns:
            List[Dict[str, Any]]: List of new trades
        """
        raise NotImplementedError("Subclasses must implement get_new_trades()")


class CMEExchangeClient(ExchangeClient):
    """
    Exchange client for CME.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the CME exchange client.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        super().__init__(config)
        
        # CME specific configuration
        self.api_key = config.get("api_key")
        self.api_secret = config.get("api_secret")
        self.account_id = config.get("account_id")
        
        # API endpoints
        self.base_url = config.get("base_url", "https://api.cmegroup.com")
        self.endpoints = {
            "orders": "/v1/trading/orders",
            "trades": "/v1/trading/trades"
        }
        
        # Session for API requests
        self.session = requests.Session()
        if self.api_key and self.api_secret:
            self.session.headers.update({
                "X-API-KEY": self.api_key,
                "X-API-SECRET": self.api_secret
            })
        
        # State variables
        self.last_trade_id = None
        
        self.logger.info("Initialized CME exchange client")
    
    def submit_order(self, order: Dict[str, Any]) -> bool:
        """
        Submit an order to CME.
        
        Args:
            order: Order to submit
            
        Returns:
            bool: True if order was submitted successfully, False otherwise
        """
        try:
            # Build URL
            url = f"{self.base_url}{self.endpoints['orders']}"
            
            # Build order payload
            payload = {
                "account_id": self.account_id,
                "symbol": order.get("symbol"),
                "side": order.get("side"),
                "size": order.get("size"),
                "client_order_id": order.get("order_id")
            }
            
            # Add order type and price
            order_type = order.get("type", "market")
            payload["type"] = order_type
            
            if order_type.lower() == "limit":
                payload["price"] = order.get("price")
            
            # Make request
            response = self.session.post(url, json=payload, timeout=5.0)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Check if order was accepted
            if data.get("status") == "accepted":
                self.logger.info(f"Order {order.get('order_id')} submitted successfully")
                return True
            else:
                self.logger.warning(f"Order {order.get('order_id')} rejected: {data.get('reason')}")
                return False
            
        except Exception as e:
            self.logger.error(f"Failed to submit order: {str(e)}")
            return False
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order on CME.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            bool: True if order was cancelled, False otherwise
        """
        try:
            # Build URL
            url = f"{self.base_url}{self.endpoints['orders']}/{order_id}"
            
            # Make request
            response = self.session.delete(url, timeout=5.0)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Check if order was cancelled
            if data.get("status") == "cancelled":
                self.logger.info(f"Order {order_id} cancelled successfully")
                return True
            else:
                self.logger.warning(f"Failed to cancel order {order_id}: {data.get('reason')}")
                return False
            
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            return False
    
    def cancel_all_orders(self) -> bool:
        """
        Cancel all open orders on CME.
        
        Returns:
            bool: True if orders were cancelled, False otherwise
        """
        try:
            # Build URL
            url = f"{self.base_url}{self.endpoints['orders']}"
            
            # Build payload
            payload = {
                "account_id": self.account_id,
                "action": "cancel_all"
            }
            
            # Make request
            response = self.session.post(url, json=payload, timeout=5.0)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Check if orders were cancelled
            if data.get("status") == "success":
                self.logger.info(f"All orders cancelled successfully")
                return True
            else:
                self.logger.warning(f"Failed to cancel all orders: {data.get('reason')}")
                return False
            
        except Exception as e:
            self.logger.error(f"Failed to cancel all orders: {str(e)}")
            return False
    
    def get_new_trades(self) -> List[Dict[str, Any]]:
        """
        Get new trades from CME.
        
        Returns:
            List[Dict[str, Any]]: List of new trades
        """
        try:
            # Build URL
            url = f"{self.base_url}{self.endpoints['trades']}"
            
            # Build parameters
            params = {
                "account_id": self.account_id,
                "limit": 100
            }
            
            # Add last trade ID if available
            if self.last_trade_id:
                params["since_id"] = self.last_trade_id
            
            # Make request
            response = self.session.get(url, params=params, timeout=5.0)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Extract trades
            trades = data.get("trades", [])
            
            # Update last trade ID
            if trades:
                self.last_trade_id = trades[-1].get("id")
            
            return trades
            
        except Exception as e:
            self.logger.error(f"Failed to get new trades: {str(e)}")
            return []


class SimulationExchangeClient(ExchangeClient):
    """
    Exchange client for simulation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the simulation exchange client.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        super().__init__(config)
        
        # Simulation specific configuration
        self.symbols = config.get("symbols", ["ES"])
        self.tick_size = config.get("tick_size", 0.25)
        self.initial_price = config.get("initial_price", 4500.0)
        self.volatility = config.get("volatility", 0.15)
        self.drift = config.get("drift", 0.0)
        self.spread = config.get("spread", 0.5)
        self.market_impact = config.get("market_impact", 0.1)
        
        # State variables
        self.current_prices = {symbol: self.initial_price for symbol in self.symbols}
        self.order_book = {
            "bids": {},  # price -> [orders]
            "asks": {}   # price -> [orders]
        }
        self.open_orders = {}  # order_id -> order
        self.trades = []
        self.last_trade_index = 0
        
        self.logger.info("Initialized simulation exchange client")
    
    def submit_order(self, order: Dict[str, Any]) -> bool:
        """
        Submit an order to the simulation.
        
        Args:
            order: Order to submit
            
        Returns:
            bool: True if order was submitted successfully, False otherwise
        """
        try:
            # Extract order details
            order_id = order.get("order_id")
            symbol = order.get("symbol", self.symbols[0])
            side = order.get("side")
            size = order.get("size")
            order_type = order.get("type", "market")
            price = order.get("price")
            
            # Validate order
            if not order_id or not side or not size:
                self.logger.warning(f"Invalid order: {order}")
                return False
            
            # Process based on order type
            if order_type.lower() == "market":
                # Market order
                self._execute_market_order(order)
                return True
            elif order_type.lower() == "limit":
                # Limit order
                if not price:
                    self.logger.warning(f"Limit order without price: {order}")
                    return False
                
                # Add to order book
                self._add_to_order_book(order)
                
                # Store open order
                self.open_orders[order_id] = order
                
                return True
            else:
                self.logger.warning(f"Unsupported order type: {order_type}")
                return False
            
        except Exception as e:
            self.logger.error(f"Failed to submit order: {str(e)}")
            return False
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order in the simulation.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            bool: True if order was cancelled, False otherwise
        """
        try:
            # Check if order exists
            if order_id not in self.open_orders:
                self.logger.warning(f"Order {order_id} not found")
                return False
            
            # Get order
            order = self.open_orders[order_id]
            
            # Remove from order book
            self._remove_from_order_book(order)
            
            # Remove from open orders
            del self.open_orders[order_id]
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            return False
    
    def cancel_all_orders(self) -> bool:
        """
        Cancel all open orders in the simulation.
        
        Returns:
            bool: True if orders were cancelled, False otherwise
        """
        try:
            # Get order IDs
            order_ids = list(self.open_orders.keys())
            
            # Cancel each order
            for order_id in order_ids:
                self.cancel_order(order_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel all orders: {str(e)}")
            return False
    
    def get_new_trades(self) -> List[Dict[str, Any]]:
        """
        Get new trades from the simulation.
        
        Returns:
            List[Dict[str, Any]]: List of new trades
        """
        try:
            # Update prices
            self._update_prices()
            
            # Match orders
            self._match_orders()
            
            # Get new trades
            new_trades = self.trades[self.last_trade_index:]
            self.last_trade_index = len(self.trades)
            
            return new_trades
            
        except Exception as e:
            self.logger.error(f"Failed to get new trades: {str(e)}")
            return []
    
    def _update_prices(self) -> None:
        """
        Update prices in the simulation.
        """
        try:
            # Update each symbol
            for symbol in self.symbols:
                # Get current price
                current_price = self.current_prices[symbol]
                
                # Calculate price change
                dt = 1.0 / (252 * 6.5 * 3600)  # Time step in years (1 second)
                drift_component = self.drift * dt
                vol_component = self.volatility * np.sqrt(dt) * np.random.normal()
                
                # Apply price change
                new_price = current_price * (1 + drift_component + vol_component)
                
                # Round to tick size
                new_price = round(new_price / self.tick_size) * self.tick_size
                
                # Update price
                self.current_prices[symbol] = new_price
                
        except Exception as e:
            self.logger.error(f"Failed to update prices: {str(e)}")
    
    def _match_orders(self) -> None:
        """
        Match orders in the simulation.
        """
        try:
            # Process each symbol
            for symbol in self.symbols:
                # Get current price
                current_price = self.current_prices[symbol]
                
                # Calculate bid and ask prices
                bid_price = current_price - self.spread * self.tick_size / 2
                ask_price = current_price + self.spread * self.tick_size / 2
                
                # Round to tick size
                bid_price = round(bid_price / self.tick_size) * self.tick_size
                ask_price = round(ask_price / self.tick_size) * self.tick_size
                
                # Match limit orders
                self._match_limit_orders(symbol, bid_price, ask_price)
                
        except Exception as e:
            self.logger.error(f"Failed to match orders: {str(e)}")
    
    def _match_limit_orders(self, symbol: str, bid_price: float, ask_price: float) -> None:
        """
        Match limit orders in the simulation.
        
        Args:
            symbol: Symbol to match orders for
            bid_price: Current bid price
            ask_price: Current ask price
        """
        try:
            # Get order book
            bids = self.order_book["bids"]
            asks = self.order_book["asks"]
            
            # Match buy orders
            buy_prices = sorted(bids.keys(), reverse=True)
            for price in buy_prices:
                # Check if price is at or above ask price
                if price >= ask_price:
                    # Match orders at this price level
                    orders = bids[price]
                    
                    # Process each order
                    i = 0
                    while i < len(orders):
                        order = orders[i]
                        
                        # Check if order is for this symbol
                        if order.get("symbol", self.symbols[0]) == symbol:
                            # Execute order
                            self._execute_limit_order(order, ask_price)
                            
                            # Remove from order book
                            orders.pop(i)
                            
                            # Remove from open orders
                            order_id = order.get("order_id")
                            if order_id in self.open_orders:
                                del self.open_orders[order_id]
                        else:
                            i += 1
                    
                    # Remove price level if empty
                    if not orders:
                        del bids[price]
            
            # Match sell orders
            sell_prices = sorted(asks.keys())
            for price in sell_prices:
                # Check if price is at or below bid price
                if price <= bid_price:
                    # Match orders at this price level
                    orders = asks[price]
                    
                    # Process each order
                    i = 0
                    while i < len(orders):
                        order = orders[i]
                        
                        # Check if order is for this symbol
                        if order.get("symbol", self.symbols[0]) == symbol:
                            # Execute order
                            self._execute_limit_order(order, bid_price)
                            
                            # Remove from order book
                            orders.pop(i)
                            
                            # Remove from open orders
                            order_id = order.get("order_id")
                            if order_id in self.open_orders:
                                del self.open_orders[order_id]
                        else:
                            i += 1
                    
                    # Remove price level if empty
                    if not orders:
                        del asks[price]
            
        except Exception as e:
            self.logger.error(f"Failed to match limit orders for {symbol}: {str(e)}")
    
    def _execute_market_order(self, order: Dict[str, Any]) -> None:
        """
        Execute a market order in the simulation.
        
        Args:
            order: Market order to execute
        """
        try:
            # Extract order details
            symbol = order.get("symbol", self.symbols[0])
            side = order.get("side")
            size = order.get("size")
            
            # Get current price
            current_price = self.current_prices.get(symbol, self.initial_price)
            
            # Calculate execution price with market impact
            if side == "buy":
                price = current_price + self.market_impact * self.tick_size
            else:
                price = current_price - self.market_impact * self.tick_size
            
            # Round to tick size
            price = round(price / self.tick_size) * self.tick_size
            
            # Create trade
            trade = {
                "timestamp": time.time(),
                "symbol": symbol,
                "side": side,
                "price": price,
                "size": size,
                "order_id": order.get("order_id")
            }
            
            # Add to trades
            self.trades.append(trade)
            
            self.logger.debug(f"Executed market order: {side} {size} {symbol} @ {price}")
            
        except Exception as e:
            self.logger.error(f"Failed to execute market order: {str(e)}")
    
    def _execute_limit_order(self, order: Dict[str, Any], price: float) -> None:
        """
        Execute a limit order in the simulation.
        
        Args:
            order: Limit order to execute
            price: Execution price
        """
        try:
            # Extract order details
            symbol = order.get("symbol", self.symbols[0])
            side = order.get("side")
            size = order.get("size")
            
            # Create trade
            trade = {
                "timestamp": time.time(),
                "symbol": symbol,
                "side": side,
                "price": price,
                "size": size,
                "order_id": order.get("order_id")
            }
            
            # Add to trades
            self.trades.append(trade)
            
            self.logger.debug(f"Executed limit order: {side} {size} {symbol} @ {price}")
            
        except Exception as e:
            self.logger.error(f"Failed to execute limit order: {str(e)}")
    
    def _add_to_order_book(self, order: Dict[str, Any]) -> None:
        """
        Add an order to the order book.
        
        Args:
            order: Order to add
        """
        try:
            # Extract order details
            side = order.get("side")
            price = order.get("price")
            
            # Add to appropriate side of book
            if side == "buy":
                if price not in self.order_book["bids"]:
                    self.order_book["bids"][price] = []
                
                self.order_book["bids"][price].append(order)
            elif side == "sell":
                if price not in self.order_book["asks"]:
                    self.order_book["asks"][price] = []
                
                self.order_book["asks"][price].append(order)
            
        except Exception as e:
            self.logger.error(f"Failed to add order to book: {str(e)}")
    
    def _remove_from_order_book(self, order: Dict[str, Any]) -> None:
        """
        Remove an order from the order book.
        
        Args:
            order: Order to remove
        """
        try:
            # Extract order details
            order_id = order.get("order_id")
            side = order.get("side")
            price = order.get("price")
            
            # Remove from appropriate side of book
            if side == "buy" and price in self.order_book["bids"]:
                orders = self.order_book["bids"][price]
                self.order_book["bids"][price] = [o for o in orders if o.get("order_id") != order_id]
                
                # Remove price level if empty
                if not self.order_book["bids"][price]:
                    del self.order_book["bids"][price]
                    
            elif side == "sell" and price in self.order_book["asks"]:
                orders = self.order_book["asks"][price]
                self.order_book["asks"][price] = [o for o in orders if o.get("order_id") != order_id]
                
                # Remove price level if empty
                if not self.order_book["asks"][price]:
                    del self.order_book["asks"][price]
            
        except Exception as e:
            self.logger.error(f"Failed to remove order from book: {str(e)}")


class RiskManager:
    """
    Risk manager for the execution engine.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the risk manager.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__ + '.RiskManager')
        
        # Risk parameters
        self.max_position = config.get("max_position", 100)
        self.max_order_size = config.get("max_order_size", 10)
        self.max_notional = config.get("max_notional", 1000000.0)
        self.price_limit_pct = config.get("price_limit_pct", 0.05)
        
        self.logger.info("Initialized risk manager")
    
    def check_order(self, order: Dict[str, Any], positions: Dict[str, int]) -> bool:
        """
        Check if an order passes risk checks.
        
        Args:
            order: Order to check
            positions: Current positions
            
        Returns:
            bool: True if order passes risk checks, False otherwise
        """
        try:
            # Extract order details
            symbol = order.get("symbol", "unknown")
            side = order.get("side")
            size = order.get("size")
            price = order.get("price")
            
            # Check order size
            if size > self.max_order_size:
                self.logger.warning(f"Order size {size} exceeds maximum {self.max_order_size}")
                return False
            
            # Check position limit
            current_position = positions.get(symbol, 0)
            if side == "buy":
                new_position = current_position + size
            else:
                new_position = current_position - size
            
            if abs(new_position) > self.max_position:
                self.logger.warning(f"New position {new_position} exceeds maximum {self.max_position}")
                return False
            
            # Check notional value
            if price:
                notional = price * size
                if notional > self.max_notional:
                    self.logger.warning(f"Notional value {notional} exceeds maximum {self.max_notional}")
                    return False
            
            # All checks passed
            return True
            
        except Exception as e:
            self.logger.error(f"Error in risk check: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    import yaml
    
    # Load configuration
    with open("/home/ubuntu/adaptive_market_making_implementation/config/execution_params.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize execution engine
    engine = ExecutionEngine(config)
    
    # Start engine
    engine.start()
    
    try:
        # Submit some test orders
        for i in range(5):
            # Create order
            order = {
                "symbol": "ES",
                "side": "buy" if i % 2 == 0 else "sell",
                "type": "limit",
                "price": 4500.0 + (i - 2) * 0.25,
                "size": 1
            }
            
            # Submit order
            order_id = engine.submit_order(order)
            
            # Sleep
            time.sleep(1)
        
        # Run for a while
        time.sleep(10)
        
    finally:
        # Stop engine
        engine.stop()
