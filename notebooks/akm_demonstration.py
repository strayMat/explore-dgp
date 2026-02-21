# %% [markdown]
# # AKM Variance Decomposition Demonstration
#
# This notebook demonstrates the AKM (Abowd, Kramarz, and Margolis) model for decomposing the variance of worker outcomes into worker and firm components.
#
# ## Use Case
# - **Target ($y$):** Percentage of sick leave per worker $i$ and year $t$.
# - **Covariates ($X$):** Age, Sex, Yearly Revenues.
# - **Worker Fixed Effect ($\theta_i$):** Captures time-invariant worker characteristics.
# - **Firm Fixed Effect ($\psi_j$):** Captures time-invariant firm characteristics.
#
# We explore two scenarios:
# 1. **Exogenous Revenues**: Revenue is independent of worker and firm fixed effects.
# 2. **Endogenous Revenues**: Revenue depends on worker and firm fixed effects.
#
# The AKM hypothesis of movers' exogeneity is maintained in both DGPs (workers move randomly between firms).

# %%
import os
import sys

# Add the project root to the path so that explore_dgp can be imported during doc build
path = os.getcwd()
while path != os.path.dirname(path):
    if os.path.exists(os.path.join(path, "pyproject.toml")):
        sys.path.insert(0, path)
        break
    path = os.path.dirname(path)

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from explore_dgp.analysis import AKMAnalysis  # noqa: E402
from explore_dgp.dgps import AKMDGP  # noqa: E402

# Set plotting style
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## 1. Generate Data (Exogenous Case)
# We simulate 10,000 workers and 5,000 firms over 5 years.

# %%
dgp_exog = AKMDGP(n_workers=10000, n_firms=5000, n_years=5)
df_exog = dgp_exog.generate(endogeneity=False)

print(f"Dataset shape: {df_exog.shape}")
df_exog.head()

# %% [markdown]
# ## 2. Run AKM Estimation (Exogenous Case)
#
# We use `pyfixest` to estimate the worker and firm fixed effects simultaneously with the covariates.

# %%
covariates = ["age", "sex", "revenue"]
analysis_exog = AKMAnalysis(df_exog, "sick_leave", covariates)
analysis_exog.run()

decomp_exog = analysis_exog.variance_decomposition(
    true_worker_fe_col="true_worker_fe", true_firm_fe_col="true_firm_fe", true_beta=dgp_exog.params
)
print("Variance Decomposition (Exogenous Case):")
print(decomp_exog.to_string(index=False))

# %% [markdown]
# ## 3. Generate Data (Endogenous Case)
# In this scenario, `revenue` is correlated with worker and firm fixed effects.

# %%
dgp_endo = AKMDGP(n_workers=10000, n_firms=5000, n_years=5)
df_endo = dgp_endo.generate(endogeneity=True)

analysis_endo = AKMAnalysis(df_endo, "sick_leave", covariates)
analysis_endo.run()

decomp_endo = analysis_endo.variance_decomposition(
    true_worker_fe_col="true_worker_fe", true_firm_fe_col="true_firm_fe", true_beta=dgp_endo.params
)
print("\nVariance Decomposition (Endogenous Case):")
print(decomp_endo.to_string(index=False))

# %% [markdown]
# ## 4. Visualization
# We compare the variance contributions between the two cases.

# %%
decomp_exog["Case"] = "Exogenous"
decomp_endo["Case"] = "Endogenous"

df_compare = pd.concat([decomp_exog, decomp_endo])

# Filter for main variance components for the plot
plot_components = ["Var(XB)", "Var(Worker FE)", "Var(Firm FE)", "Var(Residual)"]
df_plot = df_compare[df_compare["Variance/Covariance"].isin(plot_components)]

# Create a combined label for plotting: Case + Type (Estimated/True)
df_plot["label"] = df_plot["Case"] + " (" + df_plot["Type"] + ")"

plt.figure(figsize=(14, 7))
sns.barplot(data=df_plot, x="Variance/Covariance", y="% of Var(y)", hue="label")
plt.title("Comparison of Variance Components: Estimated vs True (% of Total Variance)")
plt.ylabel("% of Var(y)")
plt.ylim(0, 100)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Discussion of Findings
#
# ### Variance Decomposition: Estimated vs True
# The AKM model allows us to decompose the variance of sick leave. By comparing the **Estimated** values (from `pyfixest`) with the **True** values (from the DGP), we can observe:
# - How well the fixed effects are captured (Worker FE and Firm FE).
# - The amount of variance attributed to covariates ($XB$).
# - The residual variance, which should be close to the true noise variance ($0.5^2 = 0.25$ in our DGP).
#
# In the output above, notice that the estimated worker and firm variances can sometimes overstate the true variances due to the "limited mobility bias" in finite samples, even if movers are exogenous.
#
# ### Endogeneity Impact
# When `revenue` is endogenous (correlated with fixed effects), we see that:
# - In the **Exogenous** case, $Cov(XB, \text{Worker FE})$ and $Cov(XB, \text{Firm FE})$ are close to zero.
# - In the **Endogenous** case, these covariances become significant, as `revenue` is explicitly generated as a function of the fixed effects.
# - The AKM model correctly identifies these relationships, as seen by the matching signs and magnitudes between Estimated and True covariances.
