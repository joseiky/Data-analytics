import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# ─── SETTINGS ────────────────────────────────────────────────────────────────
sns.set(style="whitegrid")
plt.rcParams.update({"figure.autolayout": False})

# ─── LOAD & CLEAN ────────────────────────────────────────────────────────────
df = pd.read_excel("GIT-No-toxins.xlsx")

# normalize column names
df.columns = (
    df.columns
      .str.strip()
      .str.replace(r"[\s\-]+", "_", regex=True)
      .str.lower()
)

# replace lone “N” with NA
df.replace(r"^\s*\\?n\s*$", pd.NA, regex=True, inplace=True)

# cast age, drop rows missing any key column
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df.dropna(subset=[
    "test", "result", "age",
    "specimen", "source", "gender", "ethnicity"
], inplace=True)

# bin age into decades
bins = list(range(0, 121, 10))
labels = [f"{a}-{b-1}" for a, b in zip(bins[:-1], bins[1:])]
df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

# factors to plot
factors = ["specimen", "source", "age_group", "gender", "ethnicity"]

# helper for p-value annotation
def annotate_p(ax, ctab):
    if ctab.shape[0]>1 and ctab.shape[1]>1:
        _, p, _, _ = chi2_contingency(ctab)
        txt = f"χ² p = {p:.3g}"
    else:
        txt = "χ² p = NA"
    ax.text(
        0.98, 0.95, txt,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="black", pad=2, alpha=1)
    )

# ─── FIGURE 2A: ALL-TEST FREQUENCY (2×3, legend in 6th) ───────────────────────
def plot_2a(df):
    fig, axs = plt.subplots(2, 3, figsize=(22, 12))
    axs = axs.flatten()
    handles = labels = None
    for i, factor in enumerate(factors):
        sub = df[[factor, "test"]].dropna()
        ctab = pd.crosstab(sub[factor], sub["test"])
        top = ctab.sum(axis=1).nlargest(10).index
        ctab = ctab.loc[top]
        melted = ctab.reset_index().melt(
            id_vars=factor, var_name="test", value_name="count"
        )
        ax = axs[i]
        sns.barplot(data=melted, x=factor, y="count", hue="test", ax=ax)
        ax.set_title(f"Test Frequency × {factor.title()}")
        ax.tick_params(axis="x", rotation=90)
        ax.set_ylabel("Test Count")
        annotate_p(ax, ctab)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
        ax.legend_.remove()
    # place combined legend in 6th cell
    axs[5].axis("off")
    axs[5].legend(handles, labels, title="Test Type", loc="center", fontsize=10)
    fig.suptitle("Figure 2A: Test Frequency by Factor", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("Figure2A_Test_Frequency_vs_Factors.png", dpi=400)
    plt.close(fig)

# ─── FIGURE 2B: POSITIVE-ONLY FREQUENCY (2×3, legend in 6th) ───────────────────
def plot_2b(df):
    dfp = df[df["result"]=="P"]
    fig, axs = plt.subplots(2, 3, figsize=(22, 12))
    axs = axs.flatten()
    handles = labels = None
    for i, factor in enumerate(factors):
        sub = dfp[[factor, "test"]].dropna()
        ctab = pd.crosstab(sub[factor], sub["test"])
        top = ctab.sum(axis=1).nlargest(10).index
        ctab = ctab.loc[top]
        melted = ctab.reset_index().melt(
            id_vars=factor, var_name="test", value_name="count"
        )
        ax = axs[i]
        sns.barplot(data=melted, x=factor, y="count", hue="test", ax=ax)
        ax.set_title(f"Positive Count × {factor.title()}")
        ax.tick_params(axis="x", rotation=90)
        ax.set_ylabel("Positive Count")
        annotate_p(ax, ctab)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
        ax.legend_.remove()
    axs[5].axis("off")
    axs[5].legend(handles, labels, title="Test Type", loc="center", fontsize=10)
    fig.suptitle("Figure 2B: Positive Test Frequency by Factor", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("Figure2B_Positive_Test_Frequency_vs_Factors.png", dpi=400)
    plt.close(fig)

# ─── FIGURE 2C1: HEATMAP (ENLARGED, no legend) ─────────────────────────────────
def plot_2c_heatmap(df):
    fig, axs = plt.subplots(2, 3, figsize=(33, 18))  # +50% size
    axs = axs.flatten()
    for i, factor in enumerate(factors):
        ctab = pd.crosstab(df[factor], df["test"])
        top = ctab.sum(axis=1).nlargest(10).index
        ctab = ctab.loc[top]
        sns.heatmap(
            ctab, annot=True, fmt="d", cbar=False,
            linewidths=0.5, linecolor="gray", ax=axs[i]
        )
        axs[i].set_title(f"{factor.title()} vs Pathogen Counts", fontsize=16)
        axs[i].tick_params(axis="x", rotation=90)
        axs[i].set_xlabel("Test", fontsize=14)
        axs[i].set_ylabel(factor.title(), fontsize=14)
        annotate_p(axs[i], ctab)
    axs[5].axis("off")
    fig.suptitle("Figure 2C1: Heatmap of Pathogen Counts by Factor", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("Figure2C1_Heatmap_Counts.png", dpi=400)
    plt.close(fig)

# ─── FIGURE 2C2: STACKED BAR (2×3, legend in 6th) ──────────────────────────────
def plot_2c_bar(df):
    fig, axs = plt.subplots(2, 3, figsize=(22, 12))
    axs = axs.flatten()
    handles = labels = None
    for i, factor in enumerate(factors):
        ctab = pd.crosstab(df[factor], df["test"])
        top = ctab.sum(axis=1).nlargest(10).index
        ctab = ctab.loc[top]
        ax = axs[i]
        ctab.plot(kind="bar", stacked=True, ax=ax, legend=False)
        ax.set_title(f"{factor.title()} vs Pathogen Counts", fontsize=14)
        ax.tick_params(axis="x", rotation=90)
        ax.set_ylabel("Count")
        annotate_p(ax, ctab)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    axs[5].axis("off")
    axs[5].legend(handles, labels, title="Test Type", loc="center", fontsize=10)
    fig.suptitle("Figure 2C2: Stacked Bar of Pathogen Counts by Factor", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("Figure2C2_StackedBar_Counts.png", dpi=400)
    plt.close(fig)

# ─── MAIN EXECUTION ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_2a(df)
    plot_2b(df)
    plot_2c_heatmap(df)
    plot_2c_bar(df)
