"""
Analyze Results & Produce Paper Figures
========================================
Run after run_main_experiment.py completes.

Produces:
  results/figures/fig1_scatter_uncertainty_vs_gap.pdf
  results/figures/fig2_uncertainty_by_searcher.pdf
  results/figures/fig3_bootstrap_winner_cloud.pdf  (SVM 2D)
  results/figures/fig4_correlation_heatmap.pdf
  results/tables/tab_correlations.tex
  results/tables/tab_regression.tex
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.benchmarks.tasks import get_task
from src.uncertainty.estimators import config_to_log_vec
from src.analysis.analysis import (
    compute_correlations,
    compute_correlations_by_searcher,
    linear_regression_summary,
    fig_scatter_uncertainty_vs_gap,
    fig_uncertainty_by_searcher,
    fig_bootstrap_winner_distribution,
    fig_correlation_heatmap,
    to_latex_correlation_table,
)


def load_results() -> pd.DataFrame:
    csv = Path("results/main_results.csv")
    if csv.exists():
        return pd.read_csv(csv)
    # Fallback: load from individual JSONs
    rows = []
    for f in Path("results/raw").glob("*.json"):
        with open(f) as fp:
            rows.append(json.load(fp))
    return pd.DataFrame(rows)


def main():
    fig_dir = Path("results/figures")
    tab_dir = Path("results/tables")
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    df = load_results()
    if df.empty:
        print("No results found. Run run_main_experiment.py first.")
        return

    print(f"Loaded {len(df)} results")
    print(df.describe())

    # --- Fig 1: scatter uncertainty vs gap ---
    fig_scatter_uncertainty_vs_gap(df, fig_dir / "fig1_scatter_uncertainty_vs_gap.pdf")

    # --- Fig 2: uncertainty by searcher ---
    fig_uncertainty_by_searcher(df, fig_dir / "fig2_uncertainty_by_searcher.pdf")

    # --- Fig 3: bootstrap winner cloud (SVM 2D) ---
    svm_files = list(Path("results/raw").glob("svm_rbf__breast_cancer__random_search__*.json"))
    if svm_files:
        with open(svm_files[0]) as f:
            svm_result = json.load(f)
        if "bootstrap_winners" in svm_result:
            task = get_task("svm_rbf")
            boot_vecs = np.array([
                config_to_log_vec(w, task)
                for w in svm_result["bootstrap_winners"]
            ])
            orig_vec = config_to_log_vec(svm_result["original_winner"], task)
            fig_bootstrap_winner_distribution(
                boot_vecs, orig_vec,
                param_names=list(task.search_space.keys()),
                out_path=fig_dir / "fig3_bootstrap_winner_cloud.pdf",
            )

    # --- Fig 4: correlation heatmap ---
    corr_by_searcher = compute_correlations_by_searcher(df)
    fig_correlation_heatmap(corr_by_searcher, fig_dir / "fig4_correlation_heatmap.pdf")

    # --- Tables ---
    print("\n=== Correlations (all) ===")
    corr_all = compute_correlations(df)
    print(corr_all.to_string())

    print("\n=== Correlations by searcher ===")
    print(corr_by_searcher.to_string())

    # LaTeX
    latex = to_latex_correlation_table(corr_by_searcher)
    (tab_dir / "tab_correlations.tex").write_text(latex)
    print(f"  LaTeX table saved to {tab_dir}/tab_correlations.tex")

    # Regression
    reg = linear_regression_summary(df)
    print(f"\n=== Joint Regression (R²={reg['R2']:.3f}, n={reg['n']}) ===")
    for k, v in reg["coefficients"].items():
        print(f"  {k}: {v:.4f}")

    reg_tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Joint regression of generalization gap on standardized uncertainty measures. $R^2={reg['R2']:.3f}$, $n={reg['n']}$.}}
\\label{{tab:regression}}
\\begin{{tabular}}{{lr}}
\\toprule
Predictor & Coefficient \\\\
\\midrule
Intercept & {reg['intercept']:.4f} \\\\
""" + "\n".join(
        f"{k} & {v:.4f} \\\\"
        for k, v in reg["coefficients"].items()
    ) + f"""
\\midrule
$R^2$ & {reg['R2']:.3f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    (tab_dir / "tab_regression.tex").write_text(reg_tex)
    print(f"  LaTeX table saved to {tab_dir}/tab_regression.tex")


if __name__ == "__main__":
    main()
