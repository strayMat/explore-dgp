r"""
Hand-written two-way fixed effects (TWFE) estimators, to understand what ``pyfixest`` / ``fixest`` do.

Model:

.. math::

    y_{it} = \alpha_i + \lambda_t + X_{it}\beta + \epsilon_{it}

Two equivalent ways to get :math:`\hat\beta`:

1. **LSDV** (least squares with dummy variables): put one dummy per unit and per period in the design
   matrix and run OLS. Exact but the design matrix has :math:`N + T + K` columns.
2. **Within / FWL**: by the Frisch-Waugh-Lovell theorem, :math:`\hat\beta` is the OLS coefficient of the
   regression of :math:`\tilde y` on :math:`\tilde X`, where :math:`\tilde{\cdot}` is the residual of a
   projection on the fixed effects. With two FE there is no closed form for unbalanced panels, so the
   projection is computed by **alternating projections** (demean by unit, then by time, repeat until
   convergence). This is what ``fixest``/``pyfixest`` do (with acceleration tricks).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DemeanResult:
    """Output of :func:`demean_alternating_projections`."""

    values: np.ndarray
    n_iter: int
    history: list[float] = field(default_factory=list)


def _group_mean(v: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Mean of each column of ``v`` within each group, broadcast back to observations."""
    counts = np.bincount(codes, minlength=n_groups)
    means = np.stack([np.bincount(codes, weights=v[:, k], minlength=n_groups) for k in range(v.shape[1])], axis=1)
    out: np.ndarray = (means / counts[:, None])[codes]
    return out


def demean_alternating_projections(
    values: np.ndarray, fe_codes: list[np.ndarray], tol: float = 1e-10, max_iter: int = 10_000
) -> DemeanResult:
    """
    Projects ``values`` off the space spanned by the fixed-effect dummies (method of alternating projections).

    Each sweep subtracts the group mean for each FE dimension in turn. The algorithm stops when the largest
    change in a sweep is below ``tol``. On a balanced two-way panel it converges after a single sweep.

    Args:
        values: Array of shape (n,) or (n, k).
        fe_codes: List of integer-coded FE arrays (0..G-1), one per FE dimension.
        tol: Convergence tolerance on the max absolute update.
        max_iter: Maximum number of sweeps.
    """
    v = np.asarray(values, dtype=float)
    squeeze = v.ndim == 1
    v = v.reshape(len(v), -1).copy()
    n_groups = [int(c.max()) + 1 for c in fe_codes]
    history: list[float] = []
    n_iter = 0
    for n_iter in range(1, max_iter + 1):  # noqa: B007
        v_old = v.copy()
        for codes, g in zip(fe_codes, n_groups, strict=True):
            v -= _group_mean(v, codes, g)
        delta = float(np.max(np.abs(v - v_old)))
        history.append(delta)
        if delta < tol:
            break
    return DemeanResult(values=v[:, 0] if squeeze else v, n_iter=n_iter, history=history)


def _codes(df: pd.DataFrame, cols: list[str]) -> list[np.ndarray]:
    return [pd.factorize(df[c], sort=True)[0] for c in cols]


def twfe_within(
    df: pd.DataFrame, y: str, regressors: list[str], fe: tuple[str, str] = ("unit", "time"), tol: float = 1e-10
) -> dict:
    """
    TWFE estimate of ``beta`` via demeaning (alternating projections) + OLS on demeaned data (FWL theorem).

    Returns a dict with ``coef`` (pd.Series), demeaned ``y_tilde`` / ``X_tilde``, ``n_iter`` and ``history``.
    """
    codes = _codes(df, list(fe))
    data = df[[y, *regressors]].to_numpy(dtype=float)
    res = demean_alternating_projections(data, codes, tol=tol)
    y_t, X_t = res.values[:, 0], res.values[:, 1:]
    beta, *_ = np.linalg.lstsq(X_t, y_t, rcond=None)
    return {
        "coef": pd.Series(beta, index=regressors),
        "y_tilde": y_t,
        "X_tilde": X_t,
        "n_iter": res.n_iter,
        "history": res.history,
    }


def twfe_lsdv(df: pd.DataFrame, y: str, regressors: list[str], fe: tuple[str, str] = ("unit", "time")) -> dict:
    """
    TWFE estimate by brute-force OLS on unit and time dummies (LSDV).

    One time dummy is dropped (first period) to avoid perfect collinearity: the unit dummies then play
    the role of the intercept. Returns ``coef``, ``unit_fe`` and ``time_fe`` (with ``time_fe`` of the first
    period normalised to 0).
    """
    unit, time = fe
    D_unit = pd.get_dummies(df[unit], prefix="u", dtype=float)
    D_time = pd.get_dummies(df[time], prefix="t", dtype=float).iloc[:, 1:]
    X = pd.concat([df[regressors].astype(float), D_unit, D_time], axis=1)
    beta, *_ = np.linalg.lstsq(X.to_numpy(), df[y].to_numpy(dtype=float), rcond=None)
    beta = pd.Series(beta, index=X.columns)
    k = len(regressors)
    unit_fe = pd.Series(beta.iloc[k : k + D_unit.shape[1]].to_numpy(), index=sorted(df[unit].unique()))
    time_fe = pd.Series(np.r_[0.0, beta.iloc[k + D_unit.shape[1] :].to_numpy()], index=sorted(df[time].unique()))
    return {"coef": beta.iloc[:k], "unit_fe": unit_fe, "time_fe": time_fe, "n_params": X.shape[1]}


def recover_fixed_effects(
    df: pd.DataFrame,
    y: str,
    regressors: list[str],
    coef: pd.Series,
    fe: tuple[str, str] = ("unit", "time"),
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> tuple[pd.Series, pd.Series]:
    """
    Recovers the FE levels once ``beta`` is known, by Gauss-Seidel on ``r = y - X beta``.

    FE are only identified up to a constant (``alpha_i + c`` and ``lambda_t - c`` fit equally well); we
    normalise ``lambda`` of the first period to 0, like :func:`twfe_lsdv`.
    """
    unit, time = fe
    cu, ct = _codes(df, [unit, time])
    r = df[y].to_numpy(dtype=float)
    if regressors:
        r = r - df[regressors].to_numpy(dtype=float) @ coef[regressors].to_numpy()
    nu, nt = cu.max() + 1, ct.max() + 1
    a, lam = np.zeros(nu), np.zeros(nt)
    cnt_u, cnt_t = np.bincount(cu, minlength=nu), np.bincount(ct, minlength=nt)
    for _ in range(max_iter):
        a_new = np.bincount(cu, weights=r - lam[ct], minlength=nu) / cnt_u
        lam_new = np.bincount(ct, weights=r - a_new[cu], minlength=nt) / cnt_t
        delta = max(np.max(np.abs(a_new - a)), np.max(np.abs(lam_new - lam)))
        a, lam = a_new, lam_new
        if delta < tol:
            break
    a, lam = a + lam[0], lam - lam[0]
    return (
        pd.Series(a, index=sorted(df[unit].unique())),
        pd.Series(lam, index=sorted(df[time].unique())),
    )


def twfe_treatment_weights(
    df: pd.DataFrame,
    treatment: str = "treated",
    controls: list[str] | None = None,
    fe: tuple[str, str] = ("unit", "time"),
) -> pd.Series:
    r"""
    Implicit weights of the TWFE treatment coefficient (de Chaisemartin & D'Haultfoeuille, 2020).

    With :math:`\tilde D` the residual of :math:`D` on the FE (and controls), FWL gives
    :math:`\hat\tau = \sum_{it} \tilde D_{it} y_{it} / \sum_{it} \tilde D_{it} D_{it}`. If
    :math:`y_{it} = \alpha_i + \lambda_t + x\beta + \tau_{it} D_{it} + \epsilon_{it}`, then

    .. math::

        E[\hat\tau] = \sum_{(i,t): D_{it}=1} w_{it} \tau_{it}, \quad
        w_{it} = \frac{\tilde D_{it}}{\sum_{D=1} \tilde D_{it}}

    Weights sum to 1 over treated cells but can be **negative** (typically for early adopters in late
    periods): TWFE then uses already-treated units as controls.

    Returns a Series aligned on ``df.index`` (0 for untreated observations).
    """
    codes = _codes(df, list(fe))
    D = df[treatment].to_numpy(dtype=float)
    D_t = demean_alternating_projections(D, codes).values
    if controls:
        X_t = demean_alternating_projections(df[controls].to_numpy(dtype=float), codes).values
        gamma, *_ = np.linalg.lstsq(X_t, D_t, rcond=None)
        D_t = D_t - X_t @ gamma
    w = np.where(D == 1, D_t, 0.0)
    return pd.Series(w / w.sum(), index=df.index, name="weight")


def imputation_estimator(
    df: pd.DataFrame,
    y: str = "sick_leave",
    treatment: str = "treated",
    controls: list[str] | None = None,
    fe: tuple[str, str] = ("unit", "time"),
) -> pd.Series:
    r"""
    Imputation estimator (Borusyak, Jaravel & Spiess, 2024), robust to heterogeneous effects.

    1. Fit the TWFE model **on untreated observations only** (:math:`D_{it}=0`): this learns
       :math:`\alpha_i, \lambda_t, \beta` without contamination by treatment effects.
    2. Impute the counterfactual :math:`\hat y_{it}(0)` for treated observations.
    3. :math:`\hat\tau_{it} = y_{it} - \hat y_{it}(0)`; average them as needed (overall ATT, by event time...).

    Returns :math:`\hat\tau_{it}` for treated observations (indexed like ``df``).
    """
    controls = controls or []
    unit, time = fe
    untreated = df[df[treatment] == 0]
    coef = twfe_within(untreated, y, controls, fe)["coef"] if controls else pd.Series(dtype=float)
    a, lam = recover_fixed_effects(untreated, y, controls, coef, fe)
    tr = df[df[treatment] == 1]
    y0 = a.reindex(tr[unit]).to_numpy() + lam.reindex(tr[time]).to_numpy()
    if controls:
        y0 = y0 + tr[controls].to_numpy(dtype=float) @ coef[controls].to_numpy()
    return pd.Series(tr[y].to_numpy() - y0, index=tr.index, name="tau_hat")
