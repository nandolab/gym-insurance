import logging
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from gym import Env, spaces
from gym_insurance.envs.benford import Benford
from gym_insurance.envs.utils import Wallet

# Set the random seed for reproducibility
np.random.seed(42)

# Create a custom logger
logger = logging.getLogger(__name__)

# Adjust logging levels for handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('app.log')

console_handler.setLevel(logging.INFO)  # INFO may be more appropriate for stream handler
file_handler.setLevel(logging.ERROR)    # ERROR for file logging

# Create formatters and add them to handlers
console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)
file_handler.setFormatter(file_format)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Define constants to replace magic numbers
INITIAL_WC = 0.1
WOC_MULTIPLIER = 1.10
ACTION_APPROVE = 1
ACTION_REJECT = 0

class InsurEnv(Env):
    def __init__(self, data: pd.DataFrame, index_column: str, value_column: str, state_columns: list, budget: int) -> None:
        super().__init__()
        self.index_column = index_column
        self.value_column = value_column
        self.state_columns = state_columns
        self.data = data.set_index(self.index_column)
        self.budget = Wallet(budget)
        self.episodes = 0

        # Action space: 0 = reject, 1 = approve
        self.action_space = spaces.Discrete(2)

        self._shuffle_data()

        # Formalize observation space using Gym's Box
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(len(self.state_columns) + 4,)
        )

    def _shuffle_data(self):
        self.data = shuffle(self.data, random_state=1234)
        idx = self.data.index.to_list().copy()
        np.random.shuffle(idx)
        self.sampled_idx = idx

    def _create_zero_state(self):
        # More descriptive function name for zero array creation
        return np.zeros((1, len(self.state_columns) + 4))

    def _reset(self):
        # Initialize or reset environment state
        self.steps = 0
        self.approved = 0
        self.states = []
        self.rewards = []
        self.wcs = []
        self.wocs = []
        self.done = False
        self.max = self.data[self.value_column].max()
        self._shuffle_data()

        self.current_case = self.sampled_idx[self.steps]
        self.budget = Wallet(self.budget.budget)
        self.initial_state = self.data.loc[
            self.sampled_idx[0], self.state_columns
        ].values

        return self._create_zero_state()

    def reset(self):
        return self._reset()

    @property
    def state(self):
        cols = self.state_columns
        _values = list(*self.case[cols].values)

        if self.budget.spent == 0:
            return np.array(_values + [0.0] * 4)  # More concise
        elif self.budget.pct_budget <= 1:
            idx = [k for k, value in self.budget.decisions.items() if value == 1]
            self.approved = len(idx)
            return np.array(
                _values
                + [
                    self.budget.pct_budget,
                    self.approved / max(1, self.steps),  # Avoid division by zero
                    self.wc,
                    self.woc,
                ]
            )
        else:
            return self._create_zero_state()

    @property
    def case(self):
        self.current_case = self.sampled_idx[self.steps]
        _case = self.data.filter(items=[self.current_case], axis=0)

        if _case.shape[0] > 1:
            logger.debug("More than one case presented at a time.")
            raise KeyError  # Add error message for clarity
        else:
            logger.debug(f"Step: {self.steps} -- Case {_case}")  # Downgrade to debug
            return _case

    def _update_steps(self):
        self.steps += 1

    def _update_benford(self, action):
        if self.steps == 0:
            self.wc = INITIAL_WC
            self.woc = WOC_MULTIPLIER * self.wc
        elif self.approved == 0:
            self.wc = self.woc / 2
        else:
            self.wc = self.benford_dist(action)
            self.woc = self.benford_dist(action, previous=True)

        # Add proper exception or logging for NaN values in Benford update
        if np.isnan(self.wc) or np.isnan(self.woc):
            logger.error(f"Impossible to calculate reward, NaN detected for case: {self.case}")
            raise ValueError(f"NaN encountered in reward calculation for case: {self.case}")

        self.wcs.append(self.wc)
        self.wocs.append(self.woc)

    def step(self, action):
        reward = 0
        self._update_benford(action)
        self._update_steps()

        if self.steps >= len(self.sampled_idx):  # Terminate episode if steps exceed data
            logger.info("More steps than data, impossible to continue.")
            done = True
            return self._create_zero_state(), reward, done, {}

        self.budget.current_budget[self.current_case] = self.budget.balance
        vl = self.case["valor_indenizacao"].values[0]
        observation = self.state

        if vl is not None:
            if action == ACTION_APPROVE and not self.done:  # Use defined constant
                try:
                    self.budget.withdraw(vl)
                    self.budget.decisions[self.current_case] = 1
                    reward = self._reward(action, wc=self.wc, woc=self.woc)
                except ValueError:
                    if self.budget.pct_budget <= self.data[self.value_column].min():
                        logger.error("Not enough budget")
                    if len(self.states) > 1:
                        observation = self.states[-1]
                    self.budget.decisions[self.current_case] = 0
                    self.done = True
                    self.episodes += 1
            elif action == ACTION_REJECT and not self.done:  # Use defined constant
                self.budget.decisions[self.current_case] = 0
                if self.budget.pct_budget < 1:
                    reward = self._reward(action, wc=self.wc, woc=self.woc)
                else:
                    self.done = True
                    self.episodes += 1

        self.rewards.append(reward)
        self.move_state()

        return observation, reward, self.done, {}

    def move_state(self):
        self.states.append(self.state)
