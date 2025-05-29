"""
Optimal spread calculator for the adaptive market making strategy.
This module implements the game-theoretic approach to calculate optimal bid-ask spreads.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptimalSpreadCalculator:
    """
    Calculator for optimal bid-ask spreads using game-theoretic approach.
    """
    
    def __init__(self, risk_aversion: float = 1.5):
        """
        Initialize the optimal spread calculator.
        
        Args:
            risk_aversion: Risk aversion parameter
        """
        self.risk_aversion = risk_aversion
        self.logger = logging.getLogger(__name__ + '.OptimalSpreadCalculator')
        
        self.logger.info(f"Initialized optimal spread calculator with risk_aversion={risk_aversion}")
    
    def calculate_base_spread(self, volatility: float, informed_trading_prob: float) -> float:
        """
        Calculate the base optimal spread using the Avellaneda-Stoikov model.
        
        Args:
            volatility: Market volatility (annualized)
            informed_trading_prob: Probability of informed trading
            
        Returns:
            float: Base optimal half-spread in price units
        """
        # Convert annualized volatility to per-tick volatility
        # Assuming 252 trading days, 6.5 hours per day, 3600 seconds per hour
        seconds_per_year = 252 * 6.5 * 3600
        vol_per_second = volatility / np.sqrt(seconds_per_year)
        
        # Calculate base spread using Avellaneda-Stoikov formula
        # s = γ * σ² * (1 + α) / (1 - α)
        # where:
        # - γ is risk aversion
        # - σ² is variance per second
        # - α is probability of informed trading
        
        # Avoid division by zero or negative values
        alpha_adj = min(max(informed_trading_prob, 0), 0.99)
        
        base_spread = self.risk_aversion * (vol_per_second ** 2) * (1 + alpha_adj) / (1 - alpha_adj)
        
        self.logger.debug(f"Calculated base spread: {base_spread:.6f} with volatility={volatility:.6f}, informed_trading_prob={informed_trading_prob:.4f}")
        
        return base_spread
    
    def adjust_for_inventory(self, base_spread: float, inventory: float, 
                           target_inventory: float, inventory_aversion: float,
                           max_inventory: float) -> Tuple[float, float]:
        """
        Adjust the base spread for inventory management.
        
        Args:
            base_spread: Base optimal half-spread
            inventory: Current inventory
            target_inventory: Target inventory level
            inventory_aversion: Inventory aversion parameter
            max_inventory: Maximum allowed inventory
            
        Returns:
            Tuple[float, float]: (bid_half_spread, ask_half_spread)
        """
        # Normalize inventory relative to max inventory
        normalized_inventory = (inventory - target_inventory) / max_inventory if max_inventory > 0 else 0
        
        # Calculate inventory skew factor
        inventory_skew = inventory_aversion * normalized_inventory
        
        # Apply asymmetric adjustment to bid and ask spreads
        bid_half_spread = base_spread * (1 - inventory_skew)
        ask_half_spread = base_spread * (1 + inventory_skew)
        
        # Ensure spreads are positive
        bid_half_spread = max(bid_half_spread, 0.0001)
        ask_half_spread = max(ask_half_spread, 0.0001)
        
        self.logger.debug(f"Adjusted spreads for inventory={inventory}: bid_half_spread={bid_half_spread:.6f}, ask_half_spread={ask_half_spread:.6f}")
        
        return bid_half_spread, ask_half_spread
    
    def adjust_for_market_conditions(self, half_spread: float, regime_params: Dict[str, Any]) -> float:
        """
        Adjust the spread based on current market regime parameters.
        
        Args:
            half_spread: Base half-spread
            regime_params: Parameters specific to the current market regime
            
        Returns:
            float: Adjusted half-spread
        """
        # Apply regime-specific multiplier
        spread_multiplier = regime_params.get('spread_multiplier', 1.0)
        adjusted_spread = half_spread * spread_multiplier
        
        self.logger.debug(f"Adjusted spread for market conditions: {adjusted_spread:.6f} with multiplier={spread_multiplier:.2f}")
        
        return adjusted_spread
    
    def calculate_optimal_quotes(self, mid_price: float, volatility: float, 
                               informed_trading_prob: float, inventory: float,
                               target_inventory: float, inventory_aversion: float,
                               max_inventory: float, regime_params: Dict[str, Any],
                               tick_size: float) -> Tuple[float, float]:
        """
        Calculate optimal bid and ask quotes.
        
        Args:
            mid_price: Current mid price
            volatility: Market volatility (annualized)
            informed_trading_prob: Probability of informed trading
            inventory: Current inventory
            target_inventory: Target inventory level
            inventory_aversion: Inventory aversion parameter
            max_inventory: Maximum allowed inventory
            regime_params: Parameters specific to the current market regime
            tick_size: Minimum price increment
            
        Returns:
            Tuple[float, float]: (bid_price, ask_price)
        """
        # Calculate base spread
        base_spread = self.calculate_base_spread(volatility, informed_trading_prob)
        
        # Adjust for inventory
        bid_half_spread, ask_half_spread = self.adjust_for_inventory(
            base_spread, inventory, target_inventory, inventory_aversion, max_inventory
        )
        
        # Adjust for market conditions
        bid_half_spread = self.adjust_for_market_conditions(bid_half_spread, regime_params)
        ask_half_spread = self.adjust_for_market_conditions(ask_half_spread, regime_params)
        
        # Calculate quote prices
        bid_price = mid_price - bid_half_spread
        ask_price = mid_price + ask_half_spread
        
        # Round to nearest tick
        bid_price = round(bid_price / tick_size) * tick_size
        ask_price = round(ask_price / tick_size) * tick_size
        
        # Ensure bid < ask
        if bid_price >= ask_price:
            # Adjust to maintain at least one tick spread
            mid_tick = round(mid_price / tick_size) * tick_size
            bid_price = mid_tick - tick_size
            ask_price = mid_tick + tick_size
        
        self.logger.debug(f"Calculated optimal quotes: bid={bid_price:.4f}, ask={ask_price:.4f}")
        
        return bid_price, ask_price
    
    def plot_spread_surface(self, volatility_range: np.ndarray, 
                          informed_prob_range: np.ndarray,
                          save_path: Optional[str] = None) -> None:
        """
        Plot the spread surface as a function of volatility and informed trading probability.
        
        Args:
            volatility_range: Range of volatility values
            informed_prob_range: Range of informed trading probability values
            save_path: Path to save the plot or None to display
        """
        # Create meshgrid
        vol_mesh, prob_mesh = np.meshgrid(volatility_range, informed_prob_range)
        spread_mesh = np.zeros_like(vol_mesh)
        
        # Calculate spread for each point
        for i in range(vol_mesh.shape[0]):
            for j in range(vol_mesh.shape[1]):
                spread_mesh[i, j] = self.calculate_base_spread(
                    vol_mesh[i, j], prob_mesh[i, j]
                )
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface
        surface = ax.plot_surface(vol_mesh, prob_mesh, spread_mesh, cmap='viridis', alpha=0.8)
        
        # Add colorbar
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5, label='Optimal Half-Spread')
        
        # Set labels and title
        ax.set_xlabel('Volatility (annualized)')
        ax.set_ylabel('Informed Trading Probability')
        ax.set_zlabel('Optimal Half-Spread')
        ax.set_title('Optimal Spread Surface')
        
        # Add grid
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Spread surface plot saved to {save_path}")
        else:
            plt.show()


class InventoryManager:
    """
    Manager for market maker inventory control.
    """
    
    def __init__(self, target_inventory: float = 0, inventory_aversion: float = 1.0):
        """
        Initialize the inventory manager.
        
        Args:
            target_inventory: Target inventory level
            inventory_aversion: Inventory aversion parameter
        """
        self.target_inventory = target_inventory
        self.inventory_aversion = inventory_aversion
        self.logger = logging.getLogger(__name__ + '.InventoryManager')
        
        self.logger.info(f"Initialized inventory manager with target_inventory={target_inventory}, inventory_aversion={inventory_aversion}")
    
    def calculate_inventory_limits(self, regime_params: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate inventory limits based on regime parameters.
        
        Args:
            regime_params: Parameters for different market regimes
            
        Returns:
            Dict[str, float]: Dictionary of inventory limits for each regime
        """
        limits = {}
        
        for regime, params in regime_params.items():
            # Base limit scaled by regime-specific inventory aversion
            regime_aversion = params.get('inventory_aversion', 1.0)
            base_limit = 100.0  # Default base limit
            
            # Calculate limit inversely proportional to aversion
            limit = base_limit / regime_aversion
            
            limits[regime] = limit
            
            self.logger.debug(f"Calculated inventory limit for regime {regime}: {limit:.2f}")
        
        return limits
    
    def calculate_skew_factor(self, inventory: float, target: float, 
                            max_inventory: float, regime_params: Dict[str, Any]) -> float:
        """
        Calculate the quote skew factor based on current inventory.
        
        Args:
            inventory: Current inventory
            target: Target inventory level
            max_inventory: Maximum allowed inventory
            regime_params: Parameters for the current market regime
            
        Returns:
            float: Skew factor for quote adjustment
        """
        # Get regime-specific skew parameter
        base_skew = regime_params.get('skew_factor', 0.1)
        
        # Normalize inventory deviation
        normalized_deviation = (inventory - target) / max_inventory if max_inventory > 0 else 0
        
        # Calculate skew factor
        skew_factor = base_skew * normalized_deviation
        
        self.logger.debug(f"Calculated skew factor: {skew_factor:.4f} for inventory={inventory}")
        
        return skew_factor
    
    def should_hedge(self, inventory: float, target: float, 
                   max_inventory: float, hedge_threshold: float = 0.7) -> bool:
        """
        Determine if hedging is needed based on current inventory.
        
        Args:
            inventory: Current inventory
            target: Target inventory level
            max_inventory: Maximum allowed inventory
            hedge_threshold: Threshold for hedging decision
            
        Returns:
            bool: True if hedging is recommended
        """
        # Calculate absolute normalized deviation
        abs_norm_deviation = abs(inventory - target) / max_inventory if max_inventory > 0 else 0
        
        # Compare to threshold
        should_hedge = abs_norm_deviation > hedge_threshold
        
        if should_hedge:
            self.logger.info(f"Hedging recommended for inventory={inventory}, deviation={abs_norm_deviation:.4f}")
        
        return should_hedge
    
    def calculate_hedge_size(self, inventory: float, target: float, 
                           hedge_ratio: float = 0.5) -> float:
        """
        Calculate the recommended hedge size.
        
        Args:
            inventory: Current inventory
            target: Target inventory level
            hedge_ratio: Portion of deviation to hedge
            
        Returns:
            float: Recommended hedge size
        """
        # Calculate deviation from target
        deviation = inventory - target
        
        # Calculate hedge size
        hedge_size = deviation * hedge_ratio
        
        self.logger.debug(f"Calculated hedge size: {hedge_size:.2f} for inventory={inventory}")
        
        return hedge_size
    
    def plot_inventory_policy(self, inventory_range: np.ndarray,
                            regime_params: Dict[str, Any],
                            max_inventory: float = 100.0,
                            save_path: Optional[str] = None) -> None:
        """
        Plot the inventory management policy.
        
        Args:
            inventory_range: Range of inventory values
            regime_params: Parameters for different market regimes
            max_inventory: Maximum allowed inventory
            save_path: Path to save the plot or None to display
        """
        plt.figure(figsize=(12, 8))
        
        # Plot skew factor for each regime
        for regime, params in regime_params.items():
            skew_factors = []
            
            for inv in inventory_range:
                skew = self.calculate_skew_factor(inv, self.target_inventory, max_inventory, params)
                skew_factors.append(skew)
            
            plt.plot(inventory_range, skew_factors, label=f"Regime: {regime}")
        
        # Add hedge threshold lines
        hedge_threshold = 0.7
        hedge_level_upper = self.target_inventory + hedge_threshold * max_inventory
        hedge_level_lower = self.target_inventory - hedge_threshold * max_inventory
        
        plt.axvline(hedge_level_upper, color='r', linestyle='--', label='Hedge Threshold')
        plt.axvline(hedge_level_lower, color='r', linestyle='--')
        plt.axvline(self.target_inventory, color='k', linestyle='-', label='Target Inventory')
        
        # Set labels and title
        plt.xlabel('Inventory')
        plt.ylabel('Quote Skew Factor')
        plt.title('Inventory Management Policy')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Inventory policy plot saved to {save_path}")
        else:
            plt.show()


# Example usage
if __name__ == "__main__":
    import yaml
    
    # Load configuration
    with open("/home/ubuntu/adaptive_market_making_implementation/config/strategy_params.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize spread calculator
    spread_calculator = OptimalSpreadCalculator(risk_aversion=config['general']['risk_aversion'])
    
    # Example calculation
    bid_price, ask_price = spread_calculator.calculate_optimal_quotes(
        mid_price=4500.0,
        volatility=0.15,
        informed_trading_prob=0.2,
        inventory=10,
        target_inventory=0,
        inventory_aversion=1.0,
        max_inventory=100,
        regime_params=config['regime_strategies']['normal'],
        tick_size=0.25
    )
    
    print(f"Optimal quotes: bid={bid_price:.2f}, ask={ask_price:.2f}")
    
    # Plot spread surface
    volatility_range = np.linspace(0.05, 0.3, 20)
    informed_prob_range = np.linspace(0.01, 0.5, 20)
    spread_calculator.plot_spread_surface(
        volatility_range=volatility_range,
        informed_prob_range=informed_prob_range,
        save_path="/home/ubuntu/adaptive_market_making_implementation/models/spread_surface.png"
    )
    
    # Initialize inventory manager
    inventory_manager = InventoryManager(
        target_inventory=config['general']['target_inventory'],
        inventory_aversion=config['general']['inventory_aversion']
    )
    
    # Plot inventory policy
    inventory_range = np.linspace(-100, 100, 200)
    inventory_manager.plot_inventory_policy(
        inventory_range=inventory_range,
        regime_params=config['regime_strategies'],
        max_inventory=config['general']['max_position'],
        save_path="/home/ubuntu/adaptive_market_making_implementation/models/inventory_policy.png"
    )
