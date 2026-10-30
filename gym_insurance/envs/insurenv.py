import numpy as np
import warnings

# Suppress specific runtime warnings
import pandas as pd
import logging
from typing import List, Tuple
from gymnasium import Env, spaces
from gym_insurance.envs.benford import Benford
from gym_insurance.envs.utils import (
    Wallet,
)  # Assuming Budget is defined elsewhere or imported from a module


warnings.filterwarnings("ignore", category=RuntimeWarning)


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler("app.log")],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


configure_logging()
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class InsurEnv(Env):
    def __init__(
        self,
        data: pd.DataFrame,
        index_column: str,
        value_column: str,
        state_columns: List[str],
        budget: float,
    ) -> None:
        """
        Initializes the Insurance Environment.

        Args:
            data (pd.DataFrame): The data to be used in the environment.
            index_column (str): The column name to be used as the index.
            value_column (str): The column name for values.
            state_columns (List[str]): A list of column names to be used for states.
            budget (float): The initial budget for the environment.
            episodes (int, optional): The number of episodes. Defaults to 10.
        """
        metadata = {"render_modes": ["human"], "render_fps": 30}
        super().__init__()

        self._validate_inputs(data, index_column, value_column, state_columns)

        data.set_index(index_column, inplace=True)
        self._data = data
        self._index_column = index_column
        self._value_column = value_column
        self._state_columns = state_columns
        self.initial_budget = budget
        self.episodes = 0
        self.episode_data = {}
        self._reset()

    def _validate_inputs(
        self,
        data: pd.DataFrame,
        index_column: str,
        value_column: str,
        state_columns: List[str],
    ) -> None:
        """
        Validates the input parameters provided to the environment.

        Args:
            data (pd.DataFrame): The data to be used in the environment.
            index_column (str): The column name to be used as the index.
            value_column (str): The column name for values.
            state_columns (List[str]): A list of column names to be used for states.

        Raises:
            ValueError: If any of the provided column names are not found in the data.
            ValueError: If the provided data is empty.
        """
        if index_column not in data.columns or value_column not in data.columns:
            raise ValueError("Invalid column names provided.")
        if not all(column in data.columns for column in state_columns):
            raise ValueError("Invalid state_columns provided.")
        if data.empty:
            raise ValueError("Data cannot be empty.")

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        self.episodes += 1
        self.episode_data[str(self.episodes)] = self.results.copy(deep=True)
        if self.steps>1:
            df = pd.concat(self.episode_data, names=['episodes']).reset_index(level=0)
            df.to_parquet(
                f"/insurenv_results_{self.initial_budget}.parquet"
            )
        return self._reset(), {}

    def _reset(self) -> None:
        """
        Resets the environment to the initial state for a new episode.

        This method initializes various attributes of the environment, preparing it for a new episode.
        It also shuffles the data if necessary and resets the budget and other counters.
        """
        self.wcs = []
        self.wocs = []
        self.rewards = []
        self.states = []
        self.action_space = spaces.Discrete(2)
        self.steps = 0
        self.approved = 0
        self.wc = 0  # Assuming wc is an attribute that needs to be reset, replace with actual initialization as necessary
        self.woc = 0  # Assuming woc is an attribute that needs to be reset, replace with actual initialization as necessary
        self.truncated = False
        self.done = False

        # Reset or initialize other necessary attributes here

        self._shuffle_data()  # Shuffle the data if necessary
        self._budget = Wallet(self.initial_budget, self.sampled_idx)
        self._budget.reset()  # Reset the budget

        # Initialize or reset the observation space if necessary
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self._state_columns) + 4,),
            dtype=np.float32,
        )
        self.initial_state = self._get_initial_state()
        return self.initial_state

    def _get_initial_state(self):
        initial_state = np.zeros(self.observation_space.shape)
        return initial_state

    def _shuffle_data(self) -> None:
        """
        Shuffles the data used in the environment.

        This method shuffles the rows of the data DataFrame and updates the sampled index list.
        """
        # Ensure there is data to shuffle
        if self._data.empty:
            raise ValueError("Cannot shuffle empty data.")

        self._data = self._data.sample(frac=1, random_state=RANDOM_SEED).reset_index(
            drop=True
        )
        self.sampled_idx = self._data.index.to_list()

    @property
    def current_case(self):
        self.case
        return self._budget.current_case

    @property
    def current_value(self):
        return self.case[self._value_column]

    @property
    def case(self):
        self._budget.current_case = self.sampled_idx[self.steps]
        case_data = self._data.loc[self._budget.current_case]
        if len(case_data.shape) > 1:
            logger.debug("More than one case presented at a time.")
            raise KeyError
        # logger.info(f"Step: {self.steps} -- \n Case: {case_data}")
        logger.info(f"Step: {self.steps}")
        return case_data

    @property
    def state(self) -> np.array:
        """
        Computes the state representation based on the current budget and case data.

        Returns:
            np.array: The state representation.
        """
        zero_state = np.zeros(len(self._state_columns) + 4)

        if self._budget._spent == 0:
            return zero_state
        elif self._budget.budget_spent_percentage <= 1:
            approved_cases = [
                case
                for case, decision in self._budget.decisions.items()
                if decision == 1
            ]
            approved_ratio = len(approved_cases) / self.steps if self.steps else 0
            current_case_values = self.case[self._state_columns].values.flatten()
            current_case_values = np.array(current_case_values, dtype=np.float32)
            additional_state_info = [
                self._budget.budget_spent_percentage,
                approved_ratio,
                self.wc,
                self.woc,
            ]
            return np.concatenate(
                [current_case_values, additional_state_info], dtype=np.float32
            )

        else:
            return zero_state

    def _update_steps(self):
        self.steps += 1
        if np.isnan(self.wc) or np.isnan(self.woc):
            logger.debug("Impossible to calculate reward")
        self.wcs.append(self.wc)
        self.wocs.append(self.woc)

    def reward(self, action):
        return self._reward(action)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict, bool]:
        logger.info(f"starting step {self.steps}")
        return self._step(action)

    def _reward(self, action):
        r = self.current_value
        self.wc, self.woc = self.benford_dist()
        if self.wc == self.woc:
            return 0.0

        if self.steps == 0:
            return 0.0

        if action == 1 and self.wc < self.woc:
            reward = self.woc / self.wc * r
        elif action == 1 and self.wc > self.woc:
            reward = -self.wc / self.woc * r
        elif action == 0 and self.wc < self.woc:
            reward = -self.woc / self.wc * r
        elif action == 0 and self.wc > self.woc:
            reward = self.wc / self.woc * r

        if reward > 0:
            return reward -1
        elif reward < 0:
            return reward +1  
        else:
            return reward 
 
    def _step(self, action):
        # self._update_benford(action)
        # FIXME: why use this werid looking local variable at all?
        # TODO: Why is reward zero here
        reward = 0

        if self.steps >= len(self.sampled_idx):
            logger.info("More steps than data, impossible to continue.")
            return np.zeros(len(self._state_columns) + 4), reward, True, False, {}

        # TODO: how to deal with this inital state for observation?
        observation = self.state
        if self.current_value is not None:
            if action == 1 and not self.done:
                try:
                    self._budget.withdraw(self.current_value)
                    self._budget.add_decision(case=self.current_case, decision=1)
                    reward = self._reward(action)
                except ValueError:
                    if (
                        self._budget.budget_spent_percentage
                        <= self._data[self._value_column].min()
                    ):
                        logger.error("Not enough budget")
                    if self.states:
                        observation = self.states[-1]
                    self._budget.add_decision(case=self.current_case, decision=0)
                    self.truncated = False
                    self.done = True
            elif action == 0 and not self.done:
                self._budget.add_decision(case=self.current_case, decision=0)
                if self._budget.budget_spent_percentage < 1:
                    reward = self._reward(action)
                else:
                    self.done = True
                    self.truncated = False

        self.rewards.append(reward)
        self.states.append(self.state)
        if not self.done:
            self._update_steps()
        return observation, reward, self.done, self.truncated, {}

    def _get_next_state(self, action):
        next_state = self.state.copy()
        next_state[
            -1
        ] = action  # Example: set the last element of the state to the current action
        return next_state

    def render(self, mode="human") -> None:
        """
        Renders the current state of the environment.

        Args:
            mode (str, optional): The mode in which to render the environment. Defaults to 'human'.
        """
        if mode != "human":
            raise ValueError(
                "Invalid mode for rendering. Currently, only 'human' mode is supported."
            )

        print(
            f"Steps: {self.steps}, Approved: {self.approved}, Budget: {self._budget.available}, Max Value: {self.max_value}"
        )

    def benford_dist(self) -> float:
        """
        Calculates the Benford distance for a given decision.

        Args:
            decision (int): The decision for which to calculate the Benford distance.
            previous (bool, optional): If True, use previous cases for the calculation. Defaults to False.

        Returns:
            float: The calculated Benford distance.
        """
        cases = self._budget.get_approved_cases()
        sample = self._data.loc[cases, self._value_column].values
        woc = Benford(3).distance(sample)

        cases.append(self.current_case)
        sample = self._data.loc[cases, self._value_column].values
        wc = Benford(3).distance(sample)
        return wc, woc

    @property
    def results(self) -> pd.DataFrame:
        """
        Retrieves the results of the environment's interactions.

        Returns:
            pd.DataFrame: A DataFrame containing the results data.
        """
        budgets = list(self._budget.budgets.values())

        results_data = pd.DataFrame.from_dict(
            self._budget.decisions, orient="index", columns=["decision"]
        )
        results_data = results_data.loc[results_data.decision.notnull(), :]
        wcs = self.wcs
        wocs = self.wocs
        steps = range(1, self.steps + 1)
        budgets_upper_limit = 0
        if len(self.wcs) < len(self.rewards):
            budgets_upper_limit = 1
            wcs = [0] + self.wcs
            wocs = [0] + self.wocs
            steps = range(1, self.steps + 2)

        results_data["rewards"] = self.rewards
        results_data["cum_rewards"] = results_data["rewards"].cumsum()
        results_data["budget"] = budgets[0 : self.steps + budgets_upper_limit]
        results_data["budget"].ffill(inplace=True)
        results_data["budget"].fillna(self.initial_budget, inplace=True)
        results_data["initial_budget"] = self.initial_budget
        results_data["wc"] = wcs
        results_data["woc"] = wocs
        results_data["steps"] = steps
        output_data = results_data.join(self._data)
        # self.save_as_parquet(output_data)
        return output_data

    def save_as_parquet(self, data):
        data.to_parquet(f".temp/data/experiment_results/insurenv_results_{self.initial_budget}.parquet")
        return data
