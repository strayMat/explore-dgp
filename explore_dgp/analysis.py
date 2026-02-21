from typing import Any

import pandas as pd
from pyfixest.estimation import feols
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

    def get_summary_table(self, true_gamma: float | None = None) -> pd.DataFrame:
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


class AKMAnalysis:
    """
    Wrapper for AKM variance decomposition using pyfixest.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        covariates: list[str],
        worker_id_col: str = "worker_id",
        firm_id_col: str = "firm_id",
    ):
        self.df = df.copy()
        self.target = target
        self.covariates = covariates
        self.worker_id_col = worker_id_col
        self.firm_id_col = firm_id_col
        self.fit: Any = None

    def run(self) -> Any:
        """
        Runs the AKM estimation using pyfixest.
        """
        # pyfixest formula: y ~ X1 + X2 | FE1 + FE2
        formula = f"{self.target} ~ {' + '.join(self.covariates)} | {self.worker_id_col} + {self.firm_id_col}"
        self.fit = feols(formula, data=self.df)
        return self.fit

    def variance_decomposition(
        self,
        true_worker_fe_col: str | None = None,
        true_firm_fe_col: str | None = None,
        true_beta: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """
        Performs the AKM variance decomposition.

        Args:
            true_worker_fe_col: Column name for true worker fixed effects (optional).
            true_firm_fe_col: Column name for true firm fixed effects (optional).
            true_beta: Dictionary of true coefficients for covariates (optional).
        """
        if self.fit is None:
            raise ModelNotRunError

        # Filter dataframe to kept observations (handle dropped singletons/NaNs)
        if hasattr(self.fit, "_na_index") and len(self.fit._na_index) > 0:
            df_kept = self.df.drop(self.df.index[self.fit._na_index])
        else:
            df_kept = self.df

        # 1. Get coefficients for covariates (Estimated)
        beta_est = self.fit.coef()
        est_covariates = [c for c in self.covariates if c in beta_est.index]
        xb_est = df_kept[est_covariates] @ beta_est[est_covariates]

        # 2. Get fixed effects (Estimated)
        fixef_dict = self.fit.fixef()
        worker_fe_key = self.worker_id_col if self.worker_id_col in fixef_dict else f"C({self.worker_id_col})"
        firm_fe_key = self.firm_id_col if self.firm_id_col in fixef_dict else f"C({self.firm_id_col})"

        worker_fe_est = df_kept[self.worker_id_col].astype(str).map(fixef_dict[worker_fe_key]).fillna(0)
        firm_fe_est = df_kept[self.firm_id_col].astype(str).map(fixef_dict[firm_fe_key]).fillna(0)

        # 3. Get residuals (Estimated)
        epsilon_est = pd.Series(self.fit.resid(), index=df_kept.index)

        def compute_decomp(xb: pd.Series, w_fe: pd.Series, f_fe: pd.Series, eps: pd.Series) -> dict:
            var_y = df_kept[self.target].var()
            return {
                "Var(y)": var_y,
                "Var(XB)": xb.var(),
                "Var(Worker FE)": w_fe.var(),
                "Var(Firm FE)": f_fe.var(),
                "Var(Residual)": eps.var(),
                "Cov(XB, Worker FE)": xb.cov(w_fe),
                "Cov(XB, Firm FE)": xb.cov(f_fe),
                "Cov(XB, Residual)": xb.cov(eps),
                "Cov(Worker FE, Firm FE)": w_fe.cov(f_fe),
                "Cov(Worker FE, Residual)": w_fe.cov(eps),
                "Cov(Firm FE, Residual)": f_fe.cov(eps),
            }

        decomp_est = compute_decomp(xb_est, worker_fe_est, firm_fe_est, epsilon_est)

        # Prepare final table
        results = []
        for k, v in decomp_est.items():
            results.append({
                "Variance/Covariance": k,
                "Type": "Estimated",
                "Value": v,
                "% of Var(y)": v / decomp_est["Var(y)"] * 100,
            })

        # True decomposition if requested
        if true_worker_fe_col and true_firm_fe_col and true_beta:
            # Filter true_beta to include only covariates present in df_kept
            beta_true_series = pd.Series({k: v for k, v in true_beta.items() if k in df_kept.columns})
            xb_true = df_kept[beta_true_series.index] @ beta_true_series
            worker_fe_true = df_kept[true_worker_fe_col]
            firm_fe_true = df_kept[true_firm_fe_col]
            # alpha is in dgp params as 'alpha'
            alpha_true = true_beta.get("alpha", 0.0)
            epsilon_true = df_kept[self.target] - (alpha_true + xb_true + worker_fe_true + firm_fe_true)

            decomp_true = compute_decomp(xb_true, worker_fe_true, firm_fe_true, epsilon_true)
            for k, v in decomp_true.items():
                results.append({
                    "Variance/Covariance": k,
                    "Type": "True",
                    "Value": v,
                    "% of Var(y)": v / decomp_true["Var(y)"] * 100,
                })

        return pd.DataFrame(results)
