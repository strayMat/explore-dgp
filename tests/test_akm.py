import pandas as pd
import pytest

from explore_dgp.analysis import AKMAnalysis
from explore_dgp.dgps import AKMDGP


def test_akm_dgp():
    n_workers = 100
    n_firms = 50
    n_years = 3
    dgp = AKMDGP(n_workers=n_workers, n_firms=n_firms, n_years=n_years, seed=42)
    df = dgp.generate()

    assert len(df) == n_workers * n_years
    assert "worker_id" in df.columns
    assert "firm_id" in df.columns
    assert "sick_leave" in df.columns
    assert df["worker_id"].nunique() == n_workers
    assert df["firm_id"].nunique() <= n_firms


def test_akm_analysis():
    n_workers = 50
    n_firms = 20
    n_years = 5
    dgp = AKMDGP(n_workers=n_workers, n_firms=n_firms, n_years=n_years, seed=42)
    df = dgp.generate()

    covariates = ["age", "sex", "revenue"]
    analysis = AKMAnalysis(df, "sick_leave", covariates)
    analysis.run()

    decomp = analysis.variance_decomposition()

    assert isinstance(decomp, pd.DataFrame)
    assert "Value" in decomp.columns
    assert "% of Var(y)" in decomp.columns
    assert "Type" in decomp.columns

    # Filter for estimated values
    decomp_est = decomp[decomp["Type"] == "Estimated"]

    # Check if sum of variances and 2*covariances roughly equals Var(y)
    var_y = decomp_est.loc[decomp_est["Variance/Covariance"] == "Var(y)", "Value"].values[0]
    var_xb = decomp_est.loc[decomp_est["Variance/Covariance"] == "Var(XB)", "Value"].values[0]
    var_worker = decomp_est.loc[decomp_est["Variance/Covariance"] == "Var(Worker FE)", "Value"].values[0]
    var_firm = decomp_est.loc[decomp_est["Variance/Covariance"] == "Var(Firm FE)", "Value"].values[0]
    var_eps = decomp_est.loc[decomp_est["Variance/Covariance"] == "Var(Residual)", "Value"].values[0]

    cov_xb_worker = decomp_est.loc[decomp_est["Variance/Covariance"] == "Cov(XB, Worker FE)", "Value"].values[0]
    cov_xb_firm = decomp_est.loc[decomp_est["Variance/Covariance"] == "Cov(XB, Firm FE)", "Value"].values[0]
    cov_xb_eps = decomp_est.loc[decomp_est["Variance/Covariance"] == "Cov(XB, Residual)", "Value"].values[0]
    cov_worker_firm = decomp_est.loc[decomp_est["Variance/Covariance"] == "Cov(Worker FE, Firm FE)", "Value"].values[0]
    cov_worker_eps = decomp_est.loc[decomp_est["Variance/Covariance"] == "Cov(Worker FE, Residual)", "Value"].values[0]
    cov_firm_eps = decomp_est.loc[decomp_est["Variance/Covariance"] == "Cov(Firm FE, Residual)", "Value"].values[0]

    sum_comp = (
        var_xb
        + var_worker
        + var_firm
        + var_eps
        + 2 * (cov_xb_worker + cov_xb_firm + cov_xb_eps + cov_worker_firm + cov_worker_eps + cov_firm_eps)
    )

    # It should be exactly equal if we accounted for all components correctly
    assert pytest.approx(sum_comp, rel=1e-5) == var_y


def test_akm_analysis_true_values():
    n_workers = 50
    n_firms = 20
    n_years = 5
    dgp = AKMDGP(n_workers=n_workers, n_firms=n_firms, n_years=n_years, seed=42)
    df = dgp.generate()

    covariates = ["age", "sex", "revenue"]
    analysis = AKMAnalysis(df, "sick_leave", covariates)
    analysis.run()

    decomp = analysis.variance_decomposition(
        true_worker_fe_col="true_worker_fe", true_firm_fe_col="true_firm_fe", true_beta=dgp.params
    )

    assert "True" in decomp["Type"].values
    decomp_true = decomp[decomp["Type"] == "True"]
    var_y = decomp_true.loc[decomp_true["Variance/Covariance"] == "Var(y)", "Value"].values[0]
    # Check that True decomposition also sums to Var(y)
    var_xb = decomp_true.loc[decomp_true["Variance/Covariance"] == "Var(XB)", "Value"].values[0]
    var_worker = decomp_true.loc[decomp_true["Variance/Covariance"] == "Var(Worker FE)", "Value"].values[0]
    var_firm = decomp_true.loc[decomp_true["Variance/Covariance"] == "Var(Firm FE)", "Value"].values[0]
    var_eps = decomp_true.loc[decomp_true["Variance/Covariance"] == "Var(Residual)", "Value"].values[0]

    cov_xb_worker = decomp_true.loc[decomp_true["Variance/Covariance"] == "Cov(XB, Worker FE)", "Value"].values[0]
    cov_xb_firm = decomp_true.loc[decomp_true["Variance/Covariance"] == "Cov(XB, Firm FE)", "Value"].values[0]
    cov_xb_eps = decomp_true.loc[decomp_true["Variance/Covariance"] == "Cov(XB, Residual)", "Value"].values[0]
    cov_worker_firm = decomp_true.loc[decomp_true["Variance/Covariance"] == "Cov(Worker FE, Firm FE)", "Value"].values[
        0
    ]
    cov_worker_eps = decomp_true.loc[decomp_true["Variance/Covariance"] == "Cov(Worker FE, Residual)", "Value"].values[
        0
    ]
    cov_firm_eps = decomp_true.loc[decomp_true["Variance/Covariance"] == "Cov(Firm FE, Residual)", "Value"].values[0]

    sum_comp = (
        var_xb
        + var_worker
        + var_firm
        + var_eps
        + 2 * (cov_xb_worker + cov_xb_firm + cov_xb_eps + cov_worker_firm + cov_worker_eps + cov_firm_eps)
    )
    assert pytest.approx(sum_comp, rel=1e-5) == var_y
