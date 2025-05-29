"""
Bayesian model for informed trading estimation.
This module implements a Bayesian approach to estimate the probability of informed trading.
"""

import numpy as np
import pymc3 as pm
import theano.tensor as tt
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BayesianInformedTradingEstimator:
    """
    Bayesian estimator for the probability of informed trading.
    """
    
    def __init__(self, prior_alpha: float = 0.2, prior_confidence: float = 50):
        """
        Initialize the Bayesian informed trading estimator.
        
        Args:
            prior_alpha: Prior probability of informed trading
            prior_confidence: Strength of prior belief
        """
        self.prior_alpha = prior_alpha
        self.prior_confidence = prior_confidence
        self.model = None
        self.trace = None
        self.logger = logging.getLogger(__name__ + '.BayesianInformedTradingEstimator')
        
        self.logger.info(f"Initialized Bayesian estimator with prior_alpha={prior_alpha}, prior_confidence={prior_confidence}")
    
    def build_model(self, orders: np.ndarray, returns: np.ndarray) -> pm.Model:
        """
        Build the Bayesian model for informed trading estimation.
        
        Args:
            orders: Array of order directions (-1 for sell, +1 for buy)
            returns: Array of subsequent returns
            
        Returns:
            pm.Model: PyMC3 model
        """
        self.logger.info(f"Building Bayesian model with {len(orders)} observations")
        
        with pm.Model() as model:
            # Prior for probability of informed trading
            alpha = pm.Beta('alpha', 
                           alpha=self.prior_alpha * self.prior_confidence,
                           beta=(1 - self.prior_alpha) * self.prior_confidence)
            
            # Prior for return distribution parameters
            sigma_n = pm.HalfNormal('sigma_n', sigma=0.001)  # Noise volatility
            sigma_i = pm.HalfNormal('sigma_i', sigma=0.005)  # Information volatility
            
            # Latent variables for trader type
            is_informed = pm.Bernoulli('is_informed', p=alpha, shape=len(orders))
            
            # Expected returns
            mu = pm.Deterministic('mu', is_informed * orders * sigma_i)
            
            # Observed returns
            observed_returns = pm.Normal('observed_returns', 
                                        mu=mu,
                                        sigma=sigma_n,
                                        observed=returns)
            
        return model
    
    def update(self, orders: np.ndarray, returns: np.ndarray, window_size: int = 50) -> float:
        """
        Update the estimate of informed trading probability.
        
        Args:
            orders: Array of order directions (-1 for sell, +1 for buy)
            returns: Array of subsequent returns
            window_size: Number of recent observations to use
            
        Returns:
            float: Estimated probability of informed trading
        """
        # Use only the most recent observations
        if len(orders) > window_size:
            orders = orders[-window_size:]
            returns = returns[-window_size:]
        
        self.logger.info(f"Updating estimate with {len(orders)} observations")
        
        try:
            # Build model
            model = self.build_model(orders, returns)
            
            # Sample from posterior
            with model:
                trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.8, 
                                 return_inferencedata=False, progressbar=False)
            
            # Store results
            self.model = model
            self.trace = trace
            
            # Extract alpha estimate (mean of posterior)
            alpha_samples = trace['alpha']
            alpha_estimate = alpha_samples.mean()
            
            self.logger.info(f"Updated estimate: alpha = {alpha_estimate:.4f}")
            
            return alpha_estimate
            
        except Exception as e:
            self.logger.error(f"Error updating estimate: {str(e)}")
            # Return prior if estimation fails
            return self.prior_alpha
    
    def get_posterior_samples(self) -> Optional[np.ndarray]:
        """
        Get samples from the posterior distribution of alpha.
        
        Returns:
            Optional[np.ndarray]: Samples from posterior or None if not available
        """
        if self.trace is None:
            return None
        
        return self.trace['alpha']
    
    def plot_posterior(self, save_path: Optional[str] = None) -> None:
        """
        Plot the posterior distribution of alpha.
        
        Args:
            save_path: Path to save the plot or None to display
        """
        if self.trace is None:
            self.logger.warning("Cannot plot posterior: No trace available")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot posterior samples
        alpha_samples = self.trace['alpha']
        plt.hist(alpha_samples, bins=30, alpha=0.7, density=True)
        
        # Plot prior distribution
        import scipy.stats as stats
        x = np.linspace(0, 1, 1000)
        prior = stats.beta(self.prior_alpha * self.prior_confidence,
                          (1 - self.prior_alpha) * self.prior_confidence)
        plt.plot(x, prior.pdf(x), 'r-', label='Prior')
        
        # Add mean and credible interval
        mean_alpha = alpha_samples.mean()
        ci_lower = np.percentile(alpha_samples, 2.5)
        ci_upper = np.percentile(alpha_samples, 97.5)
        
        plt.axvline(mean_alpha, color='k', linestyle='--', label=f'Mean: {mean_alpha:.4f}')
        plt.axvline(ci_lower, color='k', linestyle=':', label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
        plt.axvline(ci_upper, color='k', linestyle=':')
        
        plt.title('Posterior Distribution of Informed Trading Probability (α)')
        plt.xlabel('α')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Posterior plot saved to {save_path}")
        else:
            plt.show()
    
    def get_informed_probability(self, orders: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """
        Get the probability that each order was informed.
        
        Args:
            orders: Array of order directions
            returns: Array of subsequent returns
            
        Returns:
            np.ndarray: Probability that each order was informed
        """
        if self.trace is None:
            self.logger.warning("Cannot get informed probabilities: No trace available")
            return np.full(len(orders), self.prior_alpha)
        
        # Extract samples of is_informed
        is_informed_samples = self.trace['is_informed']
        
        # Calculate probability for each order
        informed_probs = is_informed_samples.mean(axis=0)
        
        return informed_probs


# Example usage
if __name__ == "__main__":
    # Simulate some data
    np.random.seed(42)
    
    # Parameters
    n_samples = 500
    true_alpha = 0.3
    sigma_n = 0.001
    sigma_i = 0.003
    
    # Generate data
    is_informed = np.random.binomial(1, true_alpha, n_samples)
    orders = np.random.choice([-1, 1], n_samples)
    
    # Generate returns
    noise = np.random.normal(0, sigma_n, n_samples)
    information = is_informed * orders * sigma_i
    returns = information + noise
    
    # Initialize and update estimator
    estimator = BayesianInformedTradingEstimator(prior_alpha=0.2, prior_confidence=50)
    alpha_estimate = estimator.update(orders, returns)
    
    print(f"True alpha: {true_alpha}")
    print(f"Estimated alpha: {alpha_estimate:.4f}")
    
    # Plot posterior
    estimator.plot_posterior("/home/ubuntu/adaptive_market_making_implementation/models/informed_trading_posterior.png")
