from typing import Any

import pandas as pd
from statsmodels.stats.oaxaca import OaxacaBlinder


class ModelNotRunError(Exception):
    """Exception raised when the model has not been run yet."""


class OaxacaAnalysis:
    """
    Wrapper for Oaxaca-Blinder decomposition using statsmodels.
    """

    def __init__(self, df: pd.DataFrame, target: str, covariates: list[str], group_col: str):
        self.df = df
        self.target = target
        self.covariates = covariates
        self.group_col = group_col
        self.model: Any = None

    def run(self, hasconst: bool = False, swap: bool = False) -> Any:
        """
        Runs the Oaxaca-Blinder decomposition.

        Args:
            hasconst: If False, statsmodels will add a constant to the covariates.
            swap: If True, swaps the groups.
        """
        y = self.df[self.target]
        # OaxacaBlinder expects the bifurcate column to be in exog
        X = self.df[[*self.covariates, self.group_col]]

        # statsmodels OaxacaBlinder
        self.model = OaxacaBlinder(y, X, self.group_col, hasconst=hasconst, swap=swap)
        return self.model

    def get_summary_table(self, true_gamma: float = None) -> pd.DataFrame:
        """
        Returns a summary of the decomposition.
        
        Args:
            true_gamma: The true gamma coefficient from the data generating process.
                       If provided, it will be included in the summary.
        """
        if self.model is None:
            raise ModelNotRunError

        res = self.model.two_fold()
        params = res.params

        summary = {
            "Endowment Effect": params[0],
            "Coefficient Effect": params[1],
            "Total Difference": params[2],
        }
        
        if true_gamma is not None:
            summary["True Gamma Effect"] = true_gamma

        return pd.Series(summary).to_frame(name="Value")

    def get_coefficient_details(self) -> pd.DataFrame:
        """
        Returns detailed coefficients for both groups.
        """
        if self.model is None:
            raise ModelNotRunError

        # Using internal models to get params
        p0 = self.model._f_model.params
        p1 = self.model._s_model.params

        params = pd.DataFrame({"Group 0": p0, "Group 1": p1, "Difference": p1 - p0})
        return params
