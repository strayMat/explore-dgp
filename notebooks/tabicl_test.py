# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from sklearn.model_selection import train_test_split
from tabicl import TabICLRegressor

# %%
rng = np.random.default_rng(0)
n_samples = int(3e3)
x = rng.uniform(low=-3, high=3, size=n_samples)
X = x.reshape((n_samples, 1))


def true_y_mean(x):
    return expit(x) - 0.5 - 0.1 * x


def true_y_std(x):
    return 0.07 * np.exp(-((x - 0.5) ** 2) / 0.9)


y = rng.normal(loc=true_y_mean(x), scale=true_y_std(x))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 2, random_state=0)
# %%
tabicl = TabICLRegressor(n_estimators=1)
tabicl.fit(X_train, y_train)

alphas = [0.10, 0.5, 0.90]
quantiles = tabicl.predict(X_test, output_type="quantiles", alphas=alphas)


# %%
def plot_data_and_quantiles(X_test, y_test, quantiles):
    _, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)

    # Plot test data points
    ax.scatter(x=X_test, y=y_test, alpha=0.15, color="gray")

    # sort test points for a clean line plot (optional)
    order = np.argsort(X_test[:, 0])
    x_sorted = X_test[order, 0]

    # Plot median
    ax.plot(x_sorted, quantiles[order, 1], color="darkgreen", lw=3, label="median")

    # Plot 10-90% interval
    ax.fill_between(
        x_sorted, quantiles[order, 0], quantiles[order, 2], alpha=0.18, color="green", label="10–90% interval"
    )

    ax.legend(frameon=False)
    ax.set(xlabel="Input feature x", ylabel="Target variable y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("TabICL predicted quantiles")


plot_data_and_quantiles(X_test, y_test, quantiles)
# %%
