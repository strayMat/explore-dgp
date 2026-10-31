import numpy as np
import pytest
from pyfixest.estimation import feols

from explore_dgp.dgps import StaggeredAdoptionDGP
from explore_dgp.fixed_effects import (
    demean_alternating_projections,
    imputation_estimator,
    recover_fixed_effects,
    twfe_lsdv,
    twfe_treatment_weights,
    twfe_within,
)

REG = ["treated", "unemployment"]


@pytest.fixture
def df_unbalanced():
    return StaggeredAdoptionDGP(
        n_units=60, n_periods=10, cohorts=(3, 6), cohort_shares=(0.3, 0.3), missing_rate=0.1
    ).generate()


def test_dgp_shape_and_treatment():
    dgp = StaggeredAdoptionDGP(n_units=40, n_periods=8, cohorts=(2, 5), cohort_shares=(0.4, 0.3))
    df = dgp.generate(effect_type="dynamic")
    assert len(df) == 40 * 8
    # Treatment is absorbing: once treated, always treated
    assert (df.sort_values(["unit", "time"]).groupby("unit")["treated"].diff().fillna(0) >= 0).all()
    assert (df.loc[df.treated == 0, "true_tau"] == 0).all()


def test_dgp_invalid_args():
    with pytest.raises(ValueError):
        StaggeredAdoptionDGP(cohorts=(1, 2), cohort_shares=(0.5,))
    with pytest.raises(ValueError):
        StaggeredAdoptionDGP().generate(effect_type="foo")


def test_balanced_panel_converges_in_one_sweep():
    df = StaggeredAdoptionDGP(n_units=30, n_periods=6).generate()
    codes = [df.unit.to_numpy(), df.time.to_numpy()]
    res = demean_alternating_projections(df.sick_leave.to_numpy(), codes)
    assert res.n_iter <= 2  # one sweep + one check sweep
    # Demeaned variable has zero mean in every unit and period
    assert np.allclose(df.assign(r=res.values).groupby("unit").r.mean(), 0)
    assert np.allclose(df.assign(r=res.values).groupby("time").r.mean(), 0)


def test_within_lsdv_pyfixest_agree(df_unbalanced):
    within = twfe_within(df_unbalanced, "sick_leave", REG)
    lsdv = twfe_lsdv(df_unbalanced, "sick_leave", REG)
    pf = feols("sick_leave ~ treated + unemployment | unit + time", data=df_unbalanced).coef()
    assert within["n_iter"] > 2
    np.testing.assert_allclose(within["coef"], lsdv["coef"], atol=1e-7)
    np.testing.assert_allclose(within["coef"], pf[REG], atol=1e-6)


def test_recover_fixed_effects_matches_lsdv(df_unbalanced):
    lsdv = twfe_lsdv(df_unbalanced, "sick_leave", REG)
    a, lam = recover_fixed_effects(df_unbalanced, "sick_leave", REG, lsdv["coef"])
    np.testing.assert_allclose(a, lsdv["unit_fe"], atol=1e-6)
    np.testing.assert_allclose(lam, lsdv["time_fe"], atol=1e-6)


def test_weights_reproduce_twfe(df_unbalanced):
    df = df_unbalanced.copy()
    w = twfe_treatment_weights(df, controls=["unemployment"])
    assert np.isclose(w.sum(), 1)
    # Without noise, the TWFE coefficient is exactly the weighted sum of the true cell effects
    df["y_no_noise"] = df.true_unit_fe + df.true_time_fe + 0.4 * df.unemployment + df.true_tau
    tau_hat = twfe_within(df, "y_no_noise", REG)["coef"]["treated"]
    assert np.isclose(tau_hat, (w * df.true_tau).sum(), atol=1e-8)


def test_weights_with_dynamic_effects():
    dgp = StaggeredAdoptionDGP(n_units=200, n_periods=12, seed=1)
    df = dgp.generate(effect_type="dynamic", confounded_control=False)
    w = twfe_treatment_weights(df)
    tau_hat = twfe_within(df, "sick_leave", ["treated"])["coef"]["treated"]
    # Decomposition: E[tau_hat] = sum w * tau_it; noise is small with this sample size
    assert abs(tau_hat - (w * df.true_tau).sum()) < 0.05
    assert (w < 0).any()  # some negative weights in staggered designs


def test_imputation_recovers_att():
    dgp = StaggeredAdoptionDGP(n_units=200, n_periods=12, seed=3)
    df = dgp.generate(effect_type="dynamic")
    tau_hat = imputation_estimator(df, controls=["unemployment"])
    true_att = df.loc[df.treated == 1, "true_tau"].mean()
    tau_twfe = twfe_within(df, "sick_leave", REG)["coef"]["treated"]
    assert abs(tau_hat.mean() - true_att) < 0.1
    assert abs(tau_hat.mean() - true_att) < abs(tau_twfe - true_att) / 3
