"""
Analysis Utilities
==================
Loads results, runs regressions/correlations, produces paper-ready figures.

Central empirical question:
  Does winner-selection uncertainty predict generalization gap?
  Specifically: do WS, OSC, SE predict (L_test(ĥ) - L̂_val(ĥ))?
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path


# ---------------------------------------------------------------------------
# Plotting style — clean, publication-quality
# ---------------------------------------------------------------------------

COLORS = {
    "random_search":      "#2196F3",
    "bayesian_opt":       "#FF5722",
    "successive_halving": "#4CAF50",
}

LABELS = {
    "random_search":      "Random Search",
    "bayesian_opt":       "Bayesian Opt. (TPE)",
    "successive_halving": "Successive Halving",
    "winner_stability":   "Winner Stability (WS)",
    "eps_coverage":       "ε-Optimal Coverage (OSC)",
    "selection_entropy":  "Selection Entropy (SE)",
    "mean_winner_dist":   "Mean Winner Distance",
}

def set_style():
    plt.rcParams.update({
        "font.family":        "serif",
        "font.size":          11,
        "axes.labelsize":     12,
        "axes.titlesize":     12,
        "legend.fontsize":    10,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
    })


# ---------------------------------------------------------------------------
# Core analysis: correlations between uncertainty measures and gen gap
# ---------------------------------------------------------------------------

UNCERTAINTY_COLS = ["winner_stability", "eps_coverage", "selection_entropy", "mean_winner_dist"]


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each uncertainty measure, compute Spearman and Pearson correlation
    with generalization gap across all runs.
    """
    rows = []
    for col in UNCERTAINTY_COLS:
        if col not in df.columns:
            continue
        mask = df[col].notna() & df["gen_gap"].notna()
        x = df.loc[mask, col].values
        y = df.loc[mask, "gen_gap"].values
        if len(x) < 5:
            continue
        sp_r, sp_p = stats.spearmanr(x, y)
        pe_r, pe_p = stats.pearsonr(x, y)
        rows.append({
            "measure":          col,
            "spearman_r":       sp_r,
            "spearman_p":       sp_p,
            "pearson_r":        pe_r,
            "pearson_p":        pe_p,
            "n":                int(mask.sum()),
        })
    return pd.DataFrame(rows)


def compute_correlations_by_searcher(df: pd.DataFrame) -> pd.DataFrame:
    """Same, broken down by searcher."""
    rows = []
    for searcher in df["searcher"].unique():
        sub = df[df["searcher"] == searcher]
        corr = compute_correlations(sub)
        corr["searcher"] = searcher
        rows.append(corr)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def linear_regression_summary(df: pd.DataFrame, outcome: str = "gen_gap") -> Dict:
    """
    Regress gen_gap on all uncertainty measures jointly.
    Returns coefficients, R², and p-values.
    """
    from numpy.linalg import lstsq

    cols = [c for c in UNCERTAINTY_COLS if c in df.columns]
    mask = df[cols + [outcome]].notna().all(axis=1)
    X = df.loc[mask, cols].values
    y = df.loc[mask, outcome].values

    # Standardize
    X_mean, X_std = X.mean(0), X.std(0) + 1e-8
    X_std_norm = (X - X_mean) / X_std

    # Add intercept
    X_aug = np.column_stack([np.ones(len(X_std_norm)), X_std_norm])
    coeffs, _, _, _ = lstsq(X_aug, y, rcond=None)
    y_pred = X_aug @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "intercept": coeffs[0],
        "coefficients": dict(zip(cols, coeffs[1:])),
        "R2": r2,
        "n": int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_scatter_uncertainty_vs_gap(df: pd.DataFrame, out_path: Path):
    """
    Main figure: 4 scatter plots (one per uncertainty measure) of
    uncertainty measure vs. generalization gap, coloured by searcher.
    """
    set_style()
    measures = [c for c in UNCERTAINTY_COLS if c in df.columns]
    ncols = 2
    nrows = (len(measures) + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(9, 4 * nrows))
    axes = axes.flatten()

    for ax, col in zip(axes, measures):
        for searcher, grp in df.groupby("searcher"):
            mask = grp[col].notna() & grp["gen_gap"].notna()
            ax.scatter(
                grp.loc[mask, col],
                grp.loc[mask, "gen_gap"],
                label=LABELS.get(searcher, searcher),
                color=COLORS.get(searcher, "gray"),
                alpha=0.65, s=30, linewidths=0,
            )
        # Regression line over all points
        mask = df[col].notna() & df["gen_gap"].notna()
        x = df.loc[mask, col].values
        y = df.loc[mask, "gen_gap"].values
        if len(x) >= 5:
            m, b, r, p, _ = stats.linregress(x, y)
            xline = np.linspace(x.min(), x.max(), 100)
            ax.plot(xline, m * xline + b, "k--", lw=1.5, alpha=0.7)
            sp_r, sp_p = stats.spearmanr(x, y)
            ax.set_title(f"{LABELS.get(col, col)}\nSpearman r={sp_r:.2f}, p={sp_p:.3f}")

        ax.set_xlabel(LABELS.get(col, col))
        ax.set_ylabel("Generalization Gap")
        ax.axhline(0, color="gray", lw=0.8, ls=":")

    # Remove unused axes
    for ax in axes[len(measures):]:
        ax.set_visible(False)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04), frameon=False)

    fig.suptitle("Winner-Selection Uncertainty vs. Generalization Gap", fontsize=13, y=1.01)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_uncertainty_by_searcher(df: pd.DataFrame, out_path: Path):
    """
    Box plots: distribution of each uncertainty measure by searcher.
    Tells us: does the choice of HPO algorithm affect selection reliability?
    """
    set_style()
    measures = [c for c in UNCERTAINTY_COLS if c in df.columns]
    fig, axes = plt.subplots(1, len(measures), figsize=(4 * len(measures), 4))
    if len(measures) == 1:
        axes = [axes]

    for ax, col in zip(axes, measures):
        order = sorted(df["searcher"].unique())
        data_plot = [
            df.loc[df["searcher"] == s, col].dropna().values
            for s in order
        ]
        bp = ax.boxplot(data_plot, patch_artist=True, widths=0.5)
        for patch, s in zip(bp["boxes"], order):
            patch.set_facecolor(COLORS.get(s, "gray"))
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels([LABELS.get(s, s) for s in order], rotation=20, ha="right")
        ax.set_ylabel(LABELS.get(col, col))
        ax.set_title(LABELS.get(col, col))

    fig.suptitle("Selection Uncertainty by HPO Algorithm", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_bootstrap_winner_distribution(
    bootstrap_winners_vecs: np.ndarray,  # shape (B, 2) for 2D SVM task
    original_winner_vec: np.ndarray,
    param_names: List[str],
    out_path: Path,
):
    """
    Visualize the distribution of bootstrap winners in 2D log-config space.
    Most compelling for SVM (C, gamma) — shows winner cloud.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(5, 4.5))

    ax.scatter(
        bootstrap_winners_vecs[:, 0],
        bootstrap_winners_vecs[:, 1],
        alpha=0.4, s=20, color="#2196F3", label="Bootstrap winners",
    )
    ax.scatter(
        [original_winner_vec[0]],
        [original_winner_vec[1]],
        s=120, color="#FF5722", marker="*", zorder=5,
        label="Original winner",
    )

    ax.set_xlabel(f"log({param_names[0]})")
    ax.set_ylabel(f"log({param_names[1]})")
    ax.set_title("Distribution of Bootstrap-Selected Winners\n(SVM on breast_cancer, Random Search)")
    ax.legend(frameon=False)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_correlation_heatmap(corr_df: pd.DataFrame, out_path: Path):
    """Heatmap of Spearman correlations: measure × searcher."""
    set_style()
    if corr_df.empty:
        return

    pivot = corr_df.pivot_table(
        index="measure", columns="searcher", values="spearman_r"
    )
    pivot.index = [LABELS.get(i, i) for i in pivot.index]
    pivot.columns = [LABELS.get(c, c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, ax=ax,
        linewidths=0.5, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Spearman Correlation: Uncertainty Measure vs. Generalization Gap")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# LaTeX table helpers
# ---------------------------------------------------------------------------

def to_latex_correlation_table(corr_df: pd.DataFrame) -> str:
    if corr_df.empty:
        return ""
    rows = []
    rows.append(r"\begin{table}[t]")
    rows.append(r"\centering")
    rows.append(r"\caption{Spearman correlation between selection uncertainty measures and generalization gap ($L_{\text{test}}(\hat{h}) - \hat{L}_{\text{val}}(\hat{h})$).}")
    rows.append(r"\label{tab:correlations}")
    rows.append(r"\begin{tabular}{llrr}")
    rows.append(r"\toprule")
    rows.append(r"Measure & Searcher & Spearman $r$ & $p$-value \\")
    rows.append(r"\midrule")
    for _, row in corr_df.iterrows():
        stars = ""
        if row["spearman_p"] < 0.001: stars = "***"
        elif row["spearman_p"] < 0.01: stars = "**"
        elif row["spearman_p"] < 0.05: stars = "*"
        rows.append(
            f"{LABELS.get(row['measure'], row['measure'])} & "
            f"{LABELS.get(row['searcher'], row['searcher'])} & "
            f"{row['spearman_r']:.3f}{stars} & "
            f"{row['spearman_p']:.3f} \\\\"
        )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    rows.append(r"\end{table}")
    return "\n".join(rows)
