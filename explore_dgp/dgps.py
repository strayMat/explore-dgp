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


class LinearDGP(BaseDGP):
    r"""
    Linear Data Generating Process.

    The target variable :math:`y` (percentage of sick leave) is a linear combination of covariates:

    .. math::

        y = \beta_0 + \beta_{age} \cdot age + \beta_{sex} \cdot sex + \beta_{rev} \cdot revenue + \beta_{unemp} \cdot unemp + \gamma \cdot group + \epsilon

    where :math:`\epsilon \sim \mathcal{N}(0, 1)`.
    """

    def generate(self) -> pd.DataFrame:
        df = self.generate_base_covariates()

        # Define coefficients
        self.params = {
            "beta_0": 2.0,
            "beta_age": 0.05,
            "beta_sex": 0.5,
            "beta_rev": -0.0002,
            "beta_unemp": 0.2,
            "gamma": 0.5,  # Effect of being in 2023 vs 2018
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


class NonLinearDGP(BaseDGP):
    r"""
    Non-Linear Data Generating Process.

    The target variable :math:`y` includes non-linear terms and interactions:

    .. math::

        y = \beta_0 + \beta_{age} \cdot age + \beta_{age2} \cdot age^2 + \beta_{rev} \cdot \log(revenue) + \beta_{inter} \cdot age \cdot unemp + \gamma \cdot group + \epsilon

    Oaxaca-Blinder might struggle with the linear approximation of these relationships.
    """

    def generate(self) -> pd.DataFrame:
        df = self.generate_base_covariates()

        self.params = {
            "beta_0": 5.0,
            "beta_age": 0.01,
            "beta_age2": 0.001,
            "beta_rev_log": -1.5,
            "beta_inter": 0.01,
            "gamma": 0.3,
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

    def generate(self) -> pd.DataFrame:
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
            "gamma": 0.0,  # No true group effect, all difference comes from Z and X
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
