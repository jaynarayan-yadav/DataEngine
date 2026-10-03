"""
Base class and protocol for cleaner pipeline steps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
from ..logger import CleanerLogger
from ..report import CleaningReport


class BaseCleanerStep(ABC):
    """
    Abstract base class for all data cleaning steps.
    Follows scikit-learn-like fit and transform semantics to avoid data leakage
    between train and test splits, while allowing direct one-call usage.
    """

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseCleanerStep":
        """
        Learn parameters or statistics from the DataFrame.
        """
        pass

    @abstractmethod
    def transform(
        self,
        df: pd.DataFrame,
        logger: Optional[CleanerLogger] = None,
        report: Optional[CleaningReport] = None,
    ) -> pd.DataFrame:
        """
        Apply the cleaning transformation to the DataFrame using learned parameters.
        """
        pass

    def fit_transform(
        self,
        df: pd.DataFrame,
        logger: Optional[CleanerLogger] = None,
        report: Optional[CleaningReport] = None,
    ) -> pd.DataFrame:
        """
        Fit to data, then transform it.
        """
        return self.fit(df).transform(df, logger=logger, report=report)
