from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseDGP(ABC):
    """
    Base class for Data Generating Processes.
    """

    def __init__(self, n_samples: int = 2000, seed: int = 42):
        self.n_samples = n_samples
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_base_covariates(self) -> pd.DataFrame:
        """
        Generates common covariates for the sick leave use case.

        Variables:
        - year: 2018 or 2023
        - group: 0 for 2018, 1 for 2023
        - age: worker age (18-65)
        - sex: 0 for Male, 1 for Female
        - revenue: monthly revenue in Euro
        - unemployment_rate: departmental unemployment rate (%)
        """
        group = self.rng.choice([0, 1], size=self.n_samples)
        year = np.where(group == 0, 2018, 2023)

        age = self.rng.integers(18, 65, size=self.n_samples)
        sex = self.rng.choice([0, 1], size=self.n_samples)

        # Revenue slightly higher in 2023 due to inflation/growth
        revenue_base = self.rng.normal(2500, 500, size=self.n_samples)
        revenue = np.where(year == 2023, revenue_base * 1.1, revenue_base)

        # Unemployment rate: decreasing from 2018 to 2023 as per requirements
        unemployment_rate = np.where(
            year == 2018,
            self.rng.normal(8.5, 1.0, size=self.n_samples),
            self.rng.normal(7.0, 1.0, size=self.n_samples),
        )

        return pd.DataFrame({
            "year": year,
            "group": group,
            "age": age,
            "sex": sex,
            "revenue": revenue,
            "unemployment_rate": unemployment_rate,
        })

    @abstractmethod
    def generate(self) -> pd.DataFrame:
        """Generates the full dataset including the target variable y."""
        pass


# Oaxaca Blinder DGPs
class ObLinearDGP(BaseDGP):
    r"""
    Linear Data Generating Process.

    The target variable :math:`y` (percentage of sick leave) is a linear combination of covariates:

    .. math::

        y = \beta_0 + \beta_{age} \cdot age + \beta_{sex} \cdot sex + \beta_{rev} \cdot revenue + \beta_{unemp} \cdot unemp + \gamma \cdot group + \epsilon

    where :math:`\epsilon \sim \mathcal{N}(0, 1)`.
    """

    def generate(self, true_effect: float = 0.0) -> pd.DataFrame:
        df = self.generate_base_covariates()

        # Define coefficients
        self.params = {
            "beta_0": 2.0,
            "beta_age": 0.05,
            "beta_sex": 0.5,
            "beta_rev": -0.0002,
            "beta_unemp": 0.2,
            "gamma": true_effect,  # Effect of being in 2023 vs 2018
        }

        noise = self.rng.normal(0, 0.5, size=self.n_samples)

        df["sick_leave"] = (
            self.params["beta_0"]
            + self.params["beta_age"] * df["age"]
            + self.params["beta_sex"] * df["sex"]
            + self.params["beta_rev"] * df["revenue"]
            + self.params["beta_unemp"] * df["unemployment_rate"]
            + self.params["gamma"] * df["group"]
            + noise
        )
        # Clip to ensure percentage-like behavior (though it's a toy model)
        df["sick_leave"] = df["sick_leave"].clip(lower=0)

        return df


class AKMDGP(BaseDGP):
    r"""
    Data Generating Process for AKM (Abowd, Kramarz, and Margolis) model.
    Estimates worker and firm fixed effects.

    The target variable :math:`y_{it}` (percentage of sick leave) is:

    .. math::

        y_{it} = \alpha + X_{it}\beta + \theta_i + \psi_{J(i,t)} + \epsilon_{it}

    where:
    - :math:`\theta_i` is the worker fixed effect.
    - :math:`\psi_{J(i,t)}` is the firm fixed effect for the firm :math:`J(i,t)` worker :math:`i` works for at time :math:`t`.
    """

    def __init__(
        self,
        n_workers: int = 10000,
        n_firms: int = 5000,
        n_years: int = 5,
        seed: int = 42,
    ):
        super().__init__(n_samples=n_workers * n_years, seed=seed)
        self.n_workers = n_workers
        self.n_firms = n_firms
        self.n_years = n_years

    def generate(self, endogeneity: bool = False) -> pd.DataFrame:
        """
        Generates longitudinal data for workers and firms.

        Args:
            endogeneity: If True, revenue depends on worker and firm fixed effects.
        """
        # 1. Generate IDs and Years
        worker_ids = np.repeat(np.arange(self.n_workers), self.n_years)
        years = np.tile(np.arange(self.n_years), self.n_workers)

        # 2. Worker Fixed Effects (theta_i)
        worker_fe_unique = self.rng.normal(0, 1.0, size=self.n_workers)
        worker_fe = worker_fe_unique[worker_ids]

        # 3. Firm Fixed Effects (psi_j)
        firm_fe_unique = self.rng.normal(0, 1.0, size=self.n_firms)

        # 4. Assignment of workers to firms (with mobility)
        # Each worker starts at a random firm and has a probability to move each year
        firm_ids = np.zeros(self.n_workers * self.n_years, dtype=int)

        # Initial assignment
        current_firms = self.rng.integers(0, self.n_firms, size=self.n_workers)

        move_prob = 0.2  # 20% chance to move each year to ensure enough movers

        for t in range(self.n_years):
            idx = np.arange(t, self.n_workers * self.n_years, self.n_years)
            firm_ids[idx] = current_firms

            # Decide who moves for the next year
            if t < self.n_years - 1:
                movers = self.rng.random(size=self.n_workers) < move_prob
                current_firms[movers] = self.rng.integers(0, self.n_firms, size=movers.sum())

        firm_fe = firm_fe_unique[firm_ids]

        # 5. Covariates
        # Age increases by 1 each year
        base_age = self.rng.integers(18, 60, size=self.n_workers)
        age = np.repeat(base_age, self.n_years) + years

        sex = np.repeat(self.rng.choice([0, 1], size=self.n_workers), self.n_years)

        # Yearly Revenues
        if endogeneity:
            # Revenue depends on worker and firm fixed effects
            revenue = (
                2000 + 500 * worker_fe + 300 * firm_fe + self.rng.normal(0, 200, size=self.n_workers * self.n_years)
            )
        else:
            revenue = self.rng.normal(2500, 500, size=self.n_workers * self.n_years)

        df = pd.DataFrame({
            "worker_id": worker_ids,
            "firm_id": firm_ids,
            "year": years,
            "age": age,
            "sex": sex,
            "revenue": revenue,
            "true_worker_fe": worker_fe,
            "true_firm_fe": firm_fe,
        })

        # 6. Target variable (sick leave)
        self.params = {
            "age": 0.05,
            "sex": 0.5,
            "revenue": -0.0001,
            "alpha": 2.0,
        }

        noise = self.rng.normal(0, 0.5, size=self.n_workers * self.n_years)

        df["sick_leave"] = (
            self.params["alpha"]
            + self.params["age"] * df["age"]
            + self.params["sex"] * df["sex"]
            + self.params["revenue"] * df["revenue"]
            + df["true_worker_fe"]
            + df["true_firm_fe"]
            + noise
        )

        df["sick_leave"] = df["sick_leave"].clip(lower=0)

        return df


class NonLinearDGP(BaseDGP):
    r"""
    Non-Linear Data Generating Process.

    The target variable :math:`y` includes non-linear terms and interactions:

    .. math::

        y = \beta_0 + \beta_{age} \cdot age + \beta_{age2} \cdot age^2 + \beta_{rev} \cdot \log(revenue) + \beta_{inter} \cdot age \cdot unemp + \gamma \cdot group + \epsilon

    Oaxaca-Blinder might struggle with the linear approximation of these relationships.
    """

    def generate(self, true_effect: float = 0.0) -> pd.DataFrame:
        df = self.generate_base_covariates()

        self.params = {
            "beta_0": 5.0,
            "beta_age": 0.01,
            "beta_age2": 0.001,
            "beta_rev_log": -1.5,
            "beta_inter": 0.01,
            "gamma": true_effect,  # Effect of being in 2023 vs 2018
        }

        noise = self.rng.normal(0, 0.5, size=self.n_samples)

        df["sick_leave"] = (
            self.params["beta_0"]
            + self.params["beta_age"] * df["age"]
            + self.params["beta_age2"] * (df["age"] ** 2)
            + self.params["beta_rev_log"] * np.log(df["revenue"])
            + self.params["beta_inter"] * (df["age"] * df["unemployment_rate"])
            + self.params["gamma"] * df["group"]
            + noise
        )
        df["sick_leave"] = df["sick_leave"].clip(lower=0)

        return df


class UnobservedConfounderDGP(BaseDGP):
    r"""
    Data Generating Process with an Unobserved Confounder.

    A hidden variable :math:`Z` affects both the group assignment (implicitly via year characteristics)
    and the outcome :math:`y`:

    .. math::

        Z \sim \mathcal{N}(0, 1)

    .. math::

        y = \beta_0 + X\beta + \beta_z \cdot Z + \gamma \cdot group + \epsilon

    In this scenario, :math:`Z` is correlated with the `year` but not included in the model covariates.
    Oaxaca-Blinder will attribute the effect of :math:`Z` to the group or other covariates incorrectly.

    For example, :math:`Z` could be "General Health Awareness" which increased in 2023.
    """

    def generate(self, true_effect: float = 0.0) -> pd.DataFrame:
        df = self.generate_base_covariates()

        # Unobserved confounder Z correlated with group (year)
        # Z is higher in 2023 (group 1)
        z = self.rng.normal(df["group"] * 1.0, 1.0)

        self.params = {
            "beta_0": 2.0,
            "beta_age": 0.05,
            "beta_sex": 0.5,
            "beta_rev": -0.0002,
            "beta_unemp": 0.2,
            "beta_z": 2.0,  # Strong effect of unobserved variable
            "gamma": true_effect,  # No true group effect, all difference comes from Z and X
        }

        noise = self.rng.normal(0, 0.5, size=self.n_samples)

        df["sick_leave"] = (
            self.params["beta_0"]
            + self.params["beta_age"] * df["age"]
            + self.params["beta_sex"] * df["sex"]
            + self.params["beta_rev"] * df["revenue"]
            + self.params["beta_unemp"] * df["unemployment_rate"]
            + self.params["beta_z"] * z
            + self.params["gamma"] * df["group"]
            + noise
        )
        # We don't include 'z' in the returned dataframe to simulate it being unobserved
        df["sick_leave"] = df["sick_leave"].clip(lower=0)

        return df
