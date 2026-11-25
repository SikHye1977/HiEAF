"""Export six-direction scientific plots only after all required results are complete."""
from .common import *


def export(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for stage in ("fig2", "fig3a", "fig3b"):
        checked_receipt(output / stage)
    f2, f3a, f3b = [read(output / stage / "SUMMARY.json") for stage in ("fig2", "fig3a", "fig3b")]
    expected = {(s, t, c) for s, t in DIRECTIONS for c in CONDITIONS}
    require({(r["source"], r["target"], r["condition"]) for r in f2["rows"]} == expected, "Incomplete Figure 2")
    orders = [r for r in f3a["rows"] if r["order"] in ("ordered__both", "reversed__both")]
    means = [r for r in f3b["seed_summary"] if r["cohort"] == "full" and r["method"] in VARIANTS]
    require(len(orders) == 24 and len(means) == 36, "Incomplete Figure 3")
    require(all(sorted(r["seeds"]) == list(SEEDS) for r in means), "Missing head seed")
    folder = output / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    colors = {"B": "#2368a0", "C": "#d26c28"}
    lookup = {(r["source"], r["target"], r["condition"]): r for r in f2["rows"]}
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    names = []
    for i, (s, t, c) in enumerate((s, t, c) for s, t in DIRECTIONS for c in CONDITIONS):
        r = lookup[s, t, c]
        point, lo, hi = [100 * r[k] for k in ("delta_roc_auc", "auc_ci_family_low", "auc_ci_family_high")]
        ax.plot([lo, hi], [i, i], color=colors[c], linewidth=1.8)
        ax.scatter([point], [i], color=colors[c], s=35, zorder=3)
        names.append(f"{s} → {t}  ({c})")
    ax.axvline(0, color="0.5", linestyle="--", linewidth=1)
    ax.set_yticks(range(12), names)
    ax.invert_yaxis()
    ax.set_xlabel("ROC-AUC difference (percentage points): proposed − learned score adjustment baseline")
    ax.set_title("Six-direction paired comparisons: B/C\n99.5833% individual intervals; 12-comparison Bonferroni family")
    ax.grid(axis="x", alpha=.2)
    files = []
    for ext in ("png", "pdf", "svg"):
        name = "figure2_auc_difference_6way." + ext
        fig.savefig(folder / name, dpi=300)
        files.append(name)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    labels = [f"{s} → {t}" for s, t in DIRECTIONS]
    a = {(r["source"], r["target"], r["condition"], r["order"]): r for r in orders}
    b = {(r["source"], r["target"], r["condition"], r["method"]): r for r in means}
    for column, c in enumerate(CONDITIONS):
        ax = axes[0, column]
        for offset, name, color in ((-.16, "ordered__both", "#2368a0"), (.16, "reversed__both", "#d26c28")):
            values = [100 * a[s, t, c, name]["roc_auc"] for s, t in DIRECTIONS]
            ax.scatter(values, np.arange(6) + offset, label=name.split("__")[0], color=color)
        ax.set_title(f"(a) Fixed-head order test — condition {c}")
        ax.legend(loc="best")
        ax = axes[1, column]
        for offset, name, color in ((-.2, "T0", "#999999"), (0, "T1", "#2368a0"), (.2, "T2", "#d26c28")):
            avg = [100 * b[s, t, c, name]["roc_auc"]["mean"] for s, t in DIRECTIONS]
            sd = [100 * b[s, t, c, name]["roc_auc"]["std"] for s, t in DIRECTIONS]
            ax.errorbar(avg, np.arange(6) + offset, xerr=sd, fmt="o", capsize=3, label=name, color=color)
        ax.set_title(f"(b) Temporal heads: mean ± SD, 3 seeds — condition {c}")
        ax.legend(loc="best")
    for ax in axes.flat:
        ax.set_yticks(range(6), labels)
        ax.invert_yaxis()
        ax.set_xlabel("ROC-AUC (%)")
        ax.set_xlim(0, 100)
        ax.grid(axis="x", alpha=.2)
    for ext in ("png", "pdf", "svg"):
        name = "figure3_temporal_comparison_6way." + ext
        fig.savefig(folder / name, dpi=300)
        files.append(name)
    plt.close(fig)
    source_values = dict(fig2=f2, fig3a=orders, fig3b=means,
        source_summaries={stage: dict(path=str(output / stage / "SUMMARY.json"),
                                     sha256=sha(output / stage / "SUMMARY.json")) for stage in ("fig2", "fig3a", "fig3b")})
    atomic_json(folder / "figure_source_values.json", source_values)
    files.append("figure_source_values.json")
    finish(folder, digest(source_values), files)

