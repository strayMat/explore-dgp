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
        self.results: Any = None

    def run(self, hasconst: bool = False, swap: bool = False, std: bool = True, n: int = 500) -> Any:
        """
        Runs the Oaxaca-Blinder decomposition.

        Args:
            hasconst: If False, statsmodels will add a constant to the covariates.
            swap: If True, swaps the groups.
            std: If True, compute standard errors via bootstrap.
            n: Number of bootstrap iterations.
        """
        y = self.df[self.target]
        # OaxacaBlinder expects the bifurcate column to be in exog
        X = self.df[[*self.covariates, self.group_col]]

        # statsmodels OaxacaBlinder
        self.model = OaxacaBlinder(y, X, self.group_col, hasconst=hasconst, swap=swap)
        self.results = self.model.three_fold(std=std, n=n)
        return self.model

    def get_summary_table(self) -> pd.DataFrame:
        """
        Returns a summary of the decomposition including 95% CI if available.
        """
        if self.model is None or self.results is None:
            raise ModelNotRunError

        params = self.results.params
        std = getattr(self.results, "std", None)

        data = {
            "Estimate": [params[0], params[1], params[2], params[3]],
        }

        if std is not None:
            # std usually has 3 elements: endowment, coefficient, interaction
            # We add None for the Total Difference (gap) if not available
            full_std = [*list(std), None]
            data["Std Error"] = full_std
            data["CI 95% Lower"] = [
                params[i] - 1.96 * full_std[i] if full_std[i] is not None else None for i in range(4)
            ]
            data["CI 95% Upper"] = [
                params[i] + 1.96 * full_std[i] if full_std[i] is not None else None for i in range(4)
            ]

        index = ["Endowment Effect", "Coefficient Effect", "Interaction Effect", "Total Difference"]
        return pd.DataFrame(data, index=index)

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
