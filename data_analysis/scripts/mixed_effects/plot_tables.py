from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from settings import MODELS, MODEL_NAMING

dfs = {"data_analysis/data/output/mixed_effects/p_values/results_RQ1_EN_regular_pvalues.csv": "data_analysis/plots/mixed_effects/tables/EN_regular_regression_table.png",
       "data_analysis/data/output/mixed_effects/p_values/results_RQ2_LT_regular_pvalues.csv": "data_analysis/plots/mixed_effects/tables/LT_regular_regression_table.png",
       "data_analysis/data/output/mixed_effects/p_values/results_RQ3_EN_constrained_pvalues.csv": "data_analysis/plots/mixed_effects/tables/EN_constrained_regression_table.png",
       "data_analysis/data/output/mixed_effects/p_values/results_RQ3_LT_constrained_pvalues.csv": "data_analysis/plots/mixed_effects/tables/LT_constrained_regression_table.png"}

METRIC_ORDER = ["doc_length", "dependency_distance_mean"]

METRIC_LABELS = {
    "doc_length": "Text Length",
    "dependency_distance_mean": "Mean Dependency Distance",
}

TERM_ORDER = ["(Intercept)", "levelB1", "levelC1"]
TERM_LABELS = {"(Intercept)": "(Intercept)", "levelB1": "B1", "levelC1": "C1"}
SUB_COLS = ["Est.", "SE", "p (Adj.)", "Sig."]


def fmt_p(v):
    if pd.isna(v) or v == 0 or v < 0.0001:
        return "<.0001"
    return f"{v:.4f}"


def plot_table(df_raw: pd.DataFrame, save_path: str = "table.png"):
    df = df_raw.dropna(subset=["model"]).copy()

    n_terms = len(TERM_ORDER)
    n_sub = len(SUB_COLS)
    n_models = len(MODELS)
    n_metrics = len(METRIC_ORDER)


    fixed_cols = 2
    total_cols = fixed_cols + n_terms * n_sub


    col_w = [1.4, 1.1] + [0.55, 0.55, 0.65, 0.35] * n_terms
    assert len(col_w) == total_cols


    n_data_rows = n_metrics * (1 + n_models)
    n_rows = 2 + n_data_rows

    total_w = sum(col_w)
    fig_w = total_w * 1.1
    fig_h = n_rows * 0.38

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")


    xs = [0.0]
    for w in col_w:
        xs.append(xs[-1] + w / total_w)
    cx = [(xs[i] + xs[i + 1]) / 2 for i in range(total_cols)]

    T = 0.97
    B = 0.02
    row_h = (T - B) / n_rows

    def ry(r):
        return T - (r + 0.5) * row_h

    def hline(y, lw=0.8, color="black", xstart=0, xend=1):
        ax.plot(
            [xstart, xend],
            [y, y],
            color=color,
            lw=lw,
            transform=ax.transAxes,
            clip_on=False,
        )

    def text(x, y, s, ha="center", va="center", fs=7.5, fw="normal", color="black"):
        ax.text(
            x,
            y,
            s,
            ha=ha,
            va=va,
            fontsize=fs,
            fontweight=fw,
            color=color,
            transform=ax.transAxes,
        )

    hline(T, lw=1.2)

    r = 0
    text(cx[0], ry(r), "Metric", ha="left", fs=7, color="#555")
    text(cx[1], ry(r), "Model", ha="center", fs=7, color="#555")

    for ti, term in enumerate(TERM_ORDER):
        start = fixed_cols + ti * n_sub
        end = start + n_sub - 1
        span_x = (xs[start] + xs[end + 1]) / 2
        text(span_x, ry(r), TERM_LABELS[term], fs=7.5, fw="bold")

        hline(
            ry(r) - row_h * 0.42,
            lw=0.6,
            xstart=xs[start] + 0.005,
            xend=xs[end + 1] - 0.005,
        )

    r = 1
    for ti in range(n_terms):
        for si, sc in enumerate(SUB_COLS):
            col_i = fixed_cols + ti * n_sub + si
            text(cx[col_i], ry(r), sc, fs=6.5, color="#555")

    hline(ry(1) - row_h * 0.45, lw=0.8)

    r = 2
    for metric in METRIC_ORDER:
        is_doc = metric == "doc_length"
        dec = 2 if is_doc else 4

        hline(ry(r) + row_h * 0.48, lw=0.4, color="#aaa")
        text(xs[0] + 0.005, ry(r), METRIC_LABELS[metric], ha="left", fs=7.5, fw="bold")
        r += 1

        for model in MODELS:
            text(cx[1], ry(r), MODEL_NAMING[model], ha="center", fs=7, color="#333")

            for ti, term in enumerate(TERM_ORDER):
                cell = df[
                    (df["model"] == model)
                    & (df["metric"] == metric)
                    & (df["term"] == term)
                ]

                start_col = fixed_cols + ti * n_sub

                if cell.empty:
                    text(cx[start_col], ry(r), "—", fs=7, color="#aaa")
                    continue

                row = cell.iloc[0]
                vals = [
                    f"{row['estimate']:.{dec}f}",
                    f"{row['std_error']:.{dec}f}",
                    fmt_p(row["p_value_adjusted"]),
                    str(row["significance"]) if pd.notna(row["significance"]) else "",
                ]
                colors = ["black", "#666", "#666", "#8B0000"]

                for si, (val, col) in enumerate(zip(vals, colors)):
                    col_i = start_col + si
                    text(cx[col_i], ry(r), val, fs=7, color=col)

            r += 1

    hline(ry(r - 1) - row_h * 0.48, lw=1.2)

    ax.text(
        0.5,
        1.02,
        "English [constrained student prompt]",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        transform=ax.transAxes,
    )

    plt.tight_layout(pad=0.1)
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out}")


def main():
    for csv_path, save_path in dfs.items():
        df = pd.read_csv(csv_path)
        plot_table(df, save_path)

if __name__ == "__main__":
    main()
