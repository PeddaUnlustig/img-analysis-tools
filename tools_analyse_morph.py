import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr


def plot_scatter_correlation(mutual_df, parameters):
    """
    Plot scatter plots with Pearson correlation for UNET vs SAM_05 mutual nuclei.

    Parameters:
    - mutual_df: DataFrame with columns like 'w_UNET', 'w_SAM_05', etc.
    - parameters: List of base parameter names (without _UNET or _SAM_05 suffix)
    """
    plt.figure(figsize=(15, 10))
    correlations = {}
    for i, param in enumerate(parameters):
        unet_col = f"{param}_UNET"
        sam05_col = f"{param}_SAM_05"
        plt.subplot(2, 4, i + 1)
        sns.scatterplot(x=mutual_df[unet_col], y=mutual_df[sam05_col], alpha=0.5)
        corr, p_value = pearsonr(
            mutual_df[unet_col].dropna(), mutual_df[sam05_col].dropna()
        )
        correlations[param] = corr
        plt.title(f"{param}\nCorr: {corr:.2f} (p={p_value:.3f})")
        plt.xlabel("UNET")
        plt.ylabel("SAM_05")
    plt.tight_layout()
    plt.show()

    print("Pearson Correlations (UNET vs SAM_05):")
    for param, corr in correlations.items():
        print(f"{param}: {corr:.2f}")


def plot_violin_distribution(mutual_df, parameters):
    """
    Plot violin plots comparing distributions of parameters between UNET and SAM_05.

    Parameters:
    - mutual_df: DataFrame with columns like 'w_UNET', 'w_SAM_05', etc.
    - parameters: List of base parameter names
    """
    # Melt to long format
    unet_cols = [f"{param}_UNET" for param in parameters]
    sam05_cols = [f"{param}_SAM_05" for param in parameters]
    long_df = pd.concat(
        [
            mutual_df[unet_cols]
            .melt(var_name="Parameter", value_name="Value")
            .assign(Tool="UNET"),
            mutual_df[sam05_cols]
            .melt(var_name="Parameter", value_name="Value")
            .assign(Tool="SAM_05"),
        ]
    )
    long_df["Parameter"] = long_df["Parameter"].str.replace(
        "_UNET|_SAM_05", "", regex=True
    )

    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=long_df,
        x="Parameter",
        y="Value",
        hue="Tool",
        split=False,
        inner="quartile",
    )
    plt.xticks(rotation=45)
    plt.title("Distribution of Morphological Parameters (UNET vs SAM_05)")
    plt.tight_layout()
    plt.show()

    print("Summary Statistics:")
    for param in parameters:
        unet_col = f"{param}_UNET"
        sam05_col = f"{param}_SAM_05"
        unet_stats = mutual_df[unet_col].describe()
        sam05_stats = mutual_df[sam05_col].describe()
        print(f"\n{param}:")
        print("UNET:", unet_stats[["mean", "std", "min", "max"]].to_string())
        print("SAM_05:", sam05_stats[["mean", "std", "min", "max"]].to_string())


def plot_difference_bars(mutual_df, parameters):
    """
    Plot bar plots of mean differences (UNET - SAM_05) with error bars.

    Parameters:
    - mutual_df: DataFrame with columns like 'w_UNET', 'w_SAM_05', etc.
    - parameters: List of base parameter names
    """
    differences = pd.DataFrame(
        {
            param: mutual_df[f"{param}_UNET"] - mutual_df[f"{param}_SAM_05"]
            for param in parameters
        }
    )

    mean_diff = differences.mean()
    std_diff = differences.std()

    diff_df = pd.DataFrame(
        {"Parameter": parameters, "Mean_Diff": mean_diff, "Std_Diff": std_diff}
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=diff_df,
        x="Parameter",
        y="Mean_Diff",
        yerr=diff_df["Std_Diff"],
        capsize=0.1,
    )
    plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.xticks(rotation=45)
    plt.title("Mean Difference in Morphological Parameters (UNET - SAM_05)")
    plt.ylabel("Mean Difference")
    plt.tight_layout()
    plt.show()

    print("Mean Differences (UNET - SAM_05) with Standard Deviation:")
    for param in parameters:
        print(f"{param}: Mean = {mean_diff[param]:.2f}, SD = {std_diff[param]:.2f}")


def plot_bland_altman(mutual_df, parameters):
    """
    Plot Bland-Altman plots to assess agreement between UNET and SAM_05.

    Parameters:
    - mutual_df: DataFrame with columns like 'w_UNET', 'w_SAM_05', etc.
    - parameters: List of base parameter names
    """
    plt.figure(figsize=(15, 10))
    for i, param in enumerate(parameters):
        unet_col = f"{param}_UNET"
        sam05_col = f"{param}_SAM_05"
        plt.subplot(2, 4, i + 1)
        mean_vals = (mutual_df[unet_col] + mutual_df[sam05_col]) / 2
        diff_vals = mutual_df[unet_col] - mutual_df[sam05_col]
        plt.scatter(mean_vals, diff_vals, alpha=0.5)
        mean_diff = diff_vals.mean()
        std_diff = diff_vals.std()
        plt.axhline(
            mean_diff, color="red", linestyle="--", label=f"Mean: {mean_diff:.2f}"
        )
        plt.axhline(
            mean_diff + 1.96 * std_diff,
            color="gray",
            linestyle="--",
            label="95% Limits",
        )
        plt.axhline(mean_diff - 1.96 * std_diff, color="gray", linestyle="--")
        plt.title(param)
        plt.xlabel("Mean (UNET + SAM_05)/2")
        plt.ylabel("Diff (UNET - SAM_05)")
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.show()

    print("Bland-Altman Statistics (UNET vs SAM_05):")
    for param in parameters:
        diff_vals = mutual_df[f"{param}_UNET"] - mutual_df[f"{param}_SAM_05"]
        mean_diff = diff_vals.mean()
        std_diff = diff_vals.std()
        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff
        print(
            f"{param}: Mean Diff = {mean_diff:.2f}, SD = {std_diff:.2f}, "
            f"95% LoA = [{loa_lower:.2f}, {loa_upper:.2f}]"
        )
