"""First-digit proxy benchmark. No fraud labels, global RNG state, or hidden IO."""
from __future__ import annotations

from decimal import Decimal
from typing import Sequence

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

BENFORD = np.log10(1.0 + 1.0 / np.arange(1, 10, dtype=np.float64))
REWARD_MODES = ("potential_improvement", "signed_counterfactual")


def first_digit(value: float) -> int:
    """First significant digit, including values smaller than one."""
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("amounts must be finite and strictly positive")
    return int(Decimal(str(value)).normalize().as_tuple().digits[0])


def benford_distance(counts: np.ndarray) -> float:
    """Total variation from Benford; an empty selected cohort has sentinel 1."""
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape != (9,) or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("counts must be nine finite nonnegative numbers")
    if counts.sum() == 0:
        return 1.0
    return float(np.abs(counts / counts.sum() - BENFORD).sum() / 2.0)


class InsurEnv(gym.Env):
    """Online accept/reject selection with a finite budget and explicit proxy reward.

    Input order is retained unless ``shuffle=True``. Observations expose only the
    current claim, selected-cohort frequencies, budget and progress. Features must
    already be numeric, finite and available at decision time. Input is never mutated.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, data: pd.DataFrame, budget: float,
                 feature_columns: Sequence[str] = (), amount_scale: float = 1.0,
                 reward_mode: str = "potential_improvement", max_steps: int | None = None,
                 shuffle: bool = False, render_mode: str | None = None):
        super().__init__()
        required = {"claim_id", "claim_date", "amount", *feature_columns}
        if not required.issubset(data.columns):
            raise ValueError(f"missing input columns: {sorted(required - set(data.columns))}")
        if data.empty or data["claim_id"].isna().any() or data["claim_id"].astype(str).duplicated().any():
            raise ValueError("data must have at least one row and unique non-null claim IDs")
        if data["claim_date"].isna().any():
            raise ValueError("claim dates cannot be null")
        pd.to_datetime(data["claim_date"], errors="raise")
        self._data = data.loc[:, ["claim_id", "claim_date", "amount", *feature_columns]].copy(deep=True).reset_index(drop=True)
        self._amounts = self._data["amount"].to_numpy(dtype=np.float64, copy=True)
        self._digits = np.array([first_digit(x) - 1 for x in self._amounts], dtype=np.int64)
        self._features = self._data[list(feature_columns)].to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(self._features).all():
            raise ValueError("features must be numeric and finite")
        if not np.isfinite(budget) or budget <= 0 or not np.isfinite(amount_scale) or amount_scale <= 0:
            raise ValueError("budget and amount_scale must be finite and positive")
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}")
        if max_steps is not None and (not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0):
            raise ValueError("max_steps must be a positive integer")
        if render_mode not in (None, "ansi"):
            raise ValueError("only ansi render mode is supported")
        self.initial_budget = float(budget)
        self._budget_decimal = Decimal(str(budget))
        self._amount_decimals = [Decimal(str(x)) for x in self._amounts]
        self.amount_scale = float(amount_scale)
        self.reward_mode = reward_mode
        self.max_steps = max_steps
        self.shuffle = shuffle
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(23 + len(feature_columns),), dtype=np.float32)
        self._ended = True

    @property
    def remaining_budget(self) -> float:
        return float(self._budget_decimal - self._spent_decimal)

    @property
    def distance(self) -> float:
        return benford_distance(self.counts)

    @property
    def current_amount(self) -> float:
        """The observable current amount at source precision (no future access)."""
        if self._ended:
            raise RuntimeError("reset before requesting a candidate")
        return float(self._amounts[self.order[self.position]])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        options = options or {}
        if set(options) - {"shuffle"}:
            raise ValueError("only the shuffle reset option is supported")
        self.order = np.arange(len(self._data))
        if options.get("shuffle", self.shuffle):
            self.order = self.np_random.permutation(self.order)
        self.position = 0
        self.spent = 0.0
        self._spent_decimal = Decimal(0)
        self.counts = np.zeros(9, dtype=np.int64)
        self.accepted = 0
        self.total_reward = 0.0
        self.records: list[dict] = []
        self._ended = False
        return self._observation(), self._info()

    def _observation(self):
        if self._ended or self.position >= len(self.order):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        i = self.order[self.position]
        frequencies = self.counts / self.accepted if self.accepted else np.zeros(9)
        current_digit = np.eye(9)[self._digits[i]]
        scalars = [self._amounts[i] / self.amount_scale,
                   self.remaining_budget / self.amount_scale,
                   self.remaining_budget / self.initial_budget,
                   self.position / len(self.order), self.accepted / len(self.order)]
        obs = np.concatenate([scalars, frequencies, current_digit, self._features[i]]).astype(np.float32)
        if not np.isfinite(obs).all():
            raise ValueError("observation overflows float32; rescale amounts/features")
        return obs

    def _info(self):
        return {"distance": self.distance, "remaining_budget": self.remaining_budget,
                "spent": self.spent, "accepted": self.accepted, "steps": self.position,
                "reward_mode": self.reward_mode}

    def counterfactual_delta(self) -> float:
        """Current cohort versus current claim included exactly ONCE."""
        if self._ended:
            raise RuntimeError("reset before requesting a candidate")
        i = self.order[self.position]
        if self._amount_decimals[i] > self._budget_decimal - self._spent_decimal:
            return 0.0
        candidate_counts = self.counts.copy()
        candidate_counts[self._digits[i]] += 1
        return self.distance - benford_distance(candidate_counts)

    def step(self, action):
        if self._ended:
            raise RuntimeError("episode ended; call reset before step")
        if not self.action_space.contains(action):
            raise ValueError("action must be 0 (reject) or 1 (accept)")
        action = int(action)
        i = self.order[self.position]
        amount = float(self._amounts[i])
        before = self.distance
        feasible = self._amount_decimals[i] <= self._budget_decimal - self._spent_decimal
        delta = self.counterfactual_delta()
        accepted = bool(action == 1 and feasible)
        if accepted:
            self._spent_decimal += self._amount_decimals[i]
            self.spent = float(self._spent_decimal)
            self.counts[self._digits[i]] += 1
            self.accepted += 1
        if accepted:
            reward = delta
        elif self.reward_mode == "signed_counterfactual" and feasible:
            reward = -delta
        else:
            reward = 0.0
        self.position += 1
        self.total_reward += reward
        terminated = bool(self.position >= len(self.order) or self.remaining_budget == 0)
        truncated = bool(not terminated and self.max_steps is not None and self.position >= self.max_steps)
        self._ended = terminated or truncated
        self.records.append({"claim_id": str(self._data.iloc[i]["claim_id"]),
            "step": self.position, "action": action, "accepted": accepted,
            "feasible": bool(feasible), "amount": amount,
            "first_digit": int(self._digits[i] + 1), "distance_before": before,
            "distance_after": self.distance, "counterfactual_delta": delta,
            "reward": float(reward), "remaining_budget": self.remaining_budget})
        info = self._info()
        info.update({"accepted_action": accepted, "feasible": bool(feasible), "counterfactual_delta": delta})
        return self._observation(), float(reward), terminated, truncated, info

    def render(self):
        return f"steps={self.position} accepted={self.accepted} spent={self.spent:.2f} distance={self.distance:.6f}"
