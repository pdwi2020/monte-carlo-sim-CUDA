"""
Reinforcement Learning for Hedging

This module implements RL-based hedging strategies:
- Deep hedging with policy gradient methods
- Optimal execution strategies
- Gymnasium environment for option hedging

References:
    - Buehler et al. (2019). Deep Hedging.
    - Kolm & Ritter (2019). Dynamic Replication and Hedging.
"""

import numpy as np
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass
import warnings

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    gym = None
    spaces = None
    GYM_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


@dataclass
class HedgingConfig:
    """Configuration for hedging environment."""
    S0: float = 100.0
    K: float = 100.0
    r: float = 0.05
    sigma: float = 0.2
    T: float = 1.0
    num_steps: int = 252
    transaction_cost: float = 0.001
    risk_aversion: float = 1.0
    num_options: int = -1  # -1 = short 1 option


class OptionHedgingEnv:
    """
    Gymnasium environment for option hedging.

    The agent's goal is to hedge a short option position by trading
    the underlying asset to minimize P&L variance.
    """

    def __init__(self, config: Optional[HedgingConfig] = None):
        if not GYM_AVAILABLE:
            raise ImportError("gymnasium required for RL hedging")

        self.config = config or HedgingConfig()
        self.dt = self.config.T / self.config.num_steps

        # State: [S_t, delta_t, time_to_maturity, current_position]
        self.observation_space = spaces.Box(
            low=np.array([0, -1, 0, -10]),
            high=np.array([500, 1, self.config.T, 10]),
            dtype=np.float32
        )

        # Action: hedge ratio (how many shares to hold)
        self.action_space = spaces.Box(
            low=-2.0, high=2.0, shape=(1,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.S = self.config.S0
        self.t = 0
        self.step_count = 0
        self.position = 0.0
        self.cash = 0.0
        self.pnl_history = []

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        delta = self._compute_delta()
        ttm = self.config.T - self.t
        return np.array([self.S, delta, ttm, self.position], dtype=np.float32)

    def _compute_delta(self) -> float:
        """Compute Black-Scholes delta for reference."""
        from scipy.stats import norm
        ttm = max(self.config.T - self.t, 1e-6)
        d1 = (np.log(self.S / self.config.K) +
              (self.config.r + 0.5 * self.config.sigma**2) * ttm) / \
             (self.config.sigma * np.sqrt(ttm))
        return float(norm.cdf(d1))

    def _option_payoff(self) -> float:
        """Compute option payoff at expiry."""
        return max(self.S - self.config.K, 0)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Target hedge position (number of shares)

        Returns:
            observation, reward, terminated, truncated, info
        """
        new_position = float(action[0])

        # Transaction cost
        trade_size = abs(new_position - self.position)
        cost = trade_size * self.S * self.config.transaction_cost

        # Update cash from rebalancing
        self.cash -= (new_position - self.position) * self.S + cost
        self.position = new_position

        # Simulate stock price (GBM)
        Z = self.rng.standard_normal()
        drift = (self.config.r - 0.5 * self.config.sigma**2) * self.dt
        diffusion = self.config.sigma * np.sqrt(self.dt) * Z
        self.S = self.S * np.exp(drift + diffusion)

        # Update time
        self.t += self.dt
        self.step_count += 1

        # Check if done
        terminated = self.step_count >= self.config.num_steps

        if terminated:
            # Final P&L: stock position + cash - option payoff
            final_stock_value = self.position * self.S
            option_payoff = self.config.num_options * self._option_payoff()
            total_pnl = final_stock_value + self.cash + option_payoff

            # Reward: negative of squared P&L (encourage low variance)
            reward = -self.config.risk_aversion * total_pnl**2
        else:
            # Intermediate reward: small penalty for large positions
            reward = -0.001 * abs(self.position)

        self.pnl_history.append(self.cash + self.position * self.S)

        info = {
            'stock_price': self.S,
            'position': self.position,
            'delta': self._compute_delta(),
            'cash': self.cash
        }

        return self._get_observation(), reward, terminated, False, info


class DeltaHedgeAgent:
    """Simple delta hedging agent for comparison."""

    def __init__(self, rebalance_threshold: float = 0.01):
        self.rebalance_threshold = rebalance_threshold
        self.current_position = 0.0

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Choose action based on Black-Scholes delta."""
        delta = observation[1]

        if abs(delta - self.current_position) > self.rebalance_threshold:
            self.current_position = delta

        return np.array([self.current_position], dtype=np.float32)

    def reset(self):
        self.current_position = 0.0


class SimpleNNPolicy:
    """Simple neural network policy for hedging."""

    def __init__(self, state_dim: int = 4, action_dim: int = 1):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for neural network policy")

        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Select action from policy."""
        with torch.no_grad():
            state = torch.FloatTensor(observation).unsqueeze(0)
            action = self.model(state).numpy()[0]
        return action

    def reset(self):
        pass


def run_episode(env: OptionHedgingEnv, agent, seed: Optional[int] = None) -> Dict:
    """Run a single episode and return metrics."""
    obs, _ = env.reset(seed=seed)
    agent.reset()

    total_reward = 0
    done = False

    while not done:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    # Final P&L
    final_pnl = env.cash + env.position * env.S + env.config.num_options * env._option_payoff()

    return {
        'total_reward': total_reward,
        'final_pnl': final_pnl,
        'final_position': env.position,
        'num_steps': env.step_count
    }


def evaluate_agent(
    env: OptionHedgingEnv,
    agent,
    num_episodes: int = 100,
    seed: Optional[int] = None
) -> Dict:
    """Evaluate agent over multiple episodes."""
    rng = np.random.default_rng(seed)
    results = []

    for i in range(num_episodes):
        ep_seed = rng.integers(0, 1000000)
        result = run_episode(env, agent, seed=ep_seed)
        results.append(result)

    pnls = [r['final_pnl'] for r in results]
    rewards = [r['total_reward'] for r in results]

    return {
        'mean_pnl': np.mean(pnls),
        'std_pnl': np.std(pnls),
        'mean_reward': np.mean(rewards),
        'sharpe': np.mean(pnls) / (np.std(pnls) + 1e-8),
        'num_episodes': num_episodes
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Reinforcement Learning Hedging - Demo")
    print("=" * 60)

    if not GYM_AVAILABLE:
        print("gymnasium not available. Install with: pip install gymnasium")
        exit()

    # Create environment
    config = HedgingConfig(
        S0=100.0, K=100.0, r=0.05, sigma=0.2,
        T=1.0, num_steps=52, transaction_cost=0.001
    )
    env = OptionHedgingEnv(config)

    print("\n1. Testing Delta Hedge Agent")
    print("-" * 40)
    delta_agent = DeltaHedgeAgent(rebalance_threshold=0.02)
    delta_results = evaluate_agent(env, delta_agent, num_episodes=100, seed=42)
    print(f"   Mean P&L:  ${delta_results['mean_pnl']:.2f}")
    print(f"   Std P&L:   ${delta_results['std_pnl']:.2f}")
    print(f"   Sharpe:    {delta_results['sharpe']:.3f}")

    if TORCH_AVAILABLE:
        print("\n2. Testing NN Policy (untrained)")
        print("-" * 40)
        nn_agent = SimpleNNPolicy()
        nn_results = evaluate_agent(env, nn_agent, num_episodes=100, seed=42)
        print(f"   Mean P&L:  ${nn_results['mean_pnl']:.2f}")
        print(f"   Std P&L:   ${nn_results['std_pnl']:.2f}")
        print(f"   Sharpe:    {nn_results['sharpe']:.3f}")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("For full training, use stable-baselines3:")
    print("  pip install stable-baselines3")
    print("  from stable_baselines3 import PPO")
    print("  model = PPO('MlpPolicy', env)")
    print("  model.learn(total_timesteps=100000)")
