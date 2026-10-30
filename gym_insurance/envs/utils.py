import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class Wallet:
    def __init__(self, budget: int, sample_index: list) -> None:
        self._validate_budget(budget)
        self._budget: int = budget
        self._current_case: Optional[int] = None
        self.sample_index = sample_index
        self.reset()

    def reset(self) -> None:
        self._spent = 0
        self._current_case = None
        self._current_budget: Dict[int, float] = dict.fromkeys(self.sample_index)
        self._budgets: Dict[int, float] = dict.fromkeys(self.sample_index)
        self._decisions: Dict[int, Any] = dict.fromkeys(self.sample_index)
        assert len(self._decisions.keys()) == len(self.sample_index)

    @staticmethod
    def _validate_budget(budget: int) -> None:
        if not isinstance(budget, int) or budget <= 0:
            raise ValueError(
                f"Budget should be a positive integer, not a {type(budget)}"
            )

    def withdraw(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Withdrawals should be positive")
        if value > self.balance:
            raise ValueError("Insufficient balance for withdrawal")

        self._spent += value
        if self._current_case is not None:
            self._current_budget[self._current_case] = self.budget_spent_percentage
            self._budgets[self._current_case] = self.balance
        logger.info(f"Current (%) Budget: {self.budget_spent_percentage}")

    @property
    def balance(self) -> float:
        return self._budget - self._spent

    @property
    def budget_spent_percentage(self) -> float:
        return self._spent / self._budget

    @property
    def budgets(self) -> Dict[int, float]:
        return dict(self._budgets)

    @property
    def decisions(self) -> Dict[int, Any]:
        return dict(self._decisions)

    @property
    def current_budget(self) -> Optional[float]:
        if self._current_case is None:
            return None
        return self._current_budget.get(self._current_case)

    @property
    def current_case(self) -> Optional[int]:
        return self._current_case

    @current_case.setter
    def current_case(self, value: int) -> None:
        if value is None:
            raise TypeError("This should be an index from sample_index object")
        if value < 0:
            raise ValueError("Current case cannot be negative")
        self._current_case = value

    @property
    def current_decision(self) -> Optional[Any]:
        if self._current_case is None:
            return None
        return self._decisions.get(self._current_case)

    def add_decision(self, case: int, decision: Any) -> None:
        if case < 0:
            raise ValueError("Case number cannot be negative")
        self._decisions[case] = decision

    def get_approved_cases(self) -> list:
        """
        Retrieves the case numbers of the approved cases.

        Returns:
            list: A list of case numbers for approved cases.
        """
        return list(filter(lambda case: self._decisions[case] == 1, self._decisions))

    def get_rejected_cases(self) -> list:
        """
        Retrieves the case numbers of the rejected cases.

        Returns:
            list: A list of case numbers for rejected cases.
        """
        return list(filter(lambda case: self._decisions[case] == 0, self._decisions))

    def get_approved_values(self) -> list:
        """
        Retrieves the values associated with approved cases.

        Returns:
            list: A list of values for approved cases.
        """
        return [
            value for case, value in self._budgets.items() if self._decisions[case] == 1
        ]

    def get_rejected_values(self) -> list:
        """
        Retrieves the values associated with rejected cases.

        Returns:
            list: A list of values for rejected cases.
        """
        return [
            value for case, value in self._budgets.items() if self._decisions[case] == 0
        ]

class DataDownloader:
    def __init__(self, base_dir='./temp'):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def get_xlsx_download_url(self, page_url):
        response = requests.get(page_url)
        if response.status_code != 200:
            print("Failed to retrieve the webpage")
            return None

        soup = BeautifulSoup(response.content, 'html.parser')
        download_link = soup.select_one('p > a')
        
        if download_link and 'href' in download_link.attrs:
            return download_link['href']
        else:
            return None

    def download_xlsx(self, xlsx_url, file_name):
        response = requests.get(xlsx_url)
        if response.status_code == 200:
            with open(file_name, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded {file_name}")
            return file_name
        else:
            print(f"Failed to download {xlsx_url}")
            return None

    def convert_xlsx_to_csv(self, xlsx_path, csv_path):
        df = pd.read_excel(xlsx_path)
        df.to_csv(csv_path, index=False)
        print(f"Saved as {csv_path}")

    def process_dataset(self, dataset_url):
        xlsx_url = self.get_xlsx_download_url(dataset_url)
        if xlsx_url:
            temp_xlsx_file = os.path.join(self.base_dir, xlsx_url.split('/')[-1])
            downloaded_xlsx = self.download_xlsx(xlsx_url, temp_xlsx_file)
            if downloaded_xlsx:
                temp_csv_file = temp_xlsx_file.replace('.xlsx', '.csv')
                self.convert_xlsx_to_csv(downloaded_xlsx, temp_csv_file)