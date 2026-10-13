"""Draw README figures from saved development results; no models are rerun."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "research"


def tidy(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="#e5e7eb", linewidth=0.7)
    axis.tick_params(axis="both", length=0)


def mushrooms():
    summary = json.loads((REPORTS / "mushroom_summary.json").read_text())
    default = next(row for row in summary["all_policies"]
                   if row["candidate"] == summary["primary_ranking_candidate"]
                   and row["policy"] == "default_0.5")
    rules = [summary["reference_forest_default"], default,
             summary["reference_forest_cautious"], summary["primary_high_recall_policy"]]
    labels = ["Starter forest / default", "Extra Trees / default",
              "Starter forest / cautious", "Extra Trees / cautious"]
    colors = ["#9ca3af", "#397ca5", "#9ca3af", "#397ca5"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), sharey=True)
    for axis, key, title, total in zip(
            axes, ["fn", "fp"], ["Poisonous examples missed", "Edible examples flagged"],
            [summary["poisonous_rows"], summary["edible_rows"]]):
        values = [row[key] for row in rules]
        axis.barh(range(len(rules)), values, color=colors, height=0.58)
        axis.set_xlim(0, total * 1.1)
        axis.set_title(title + " (lower is better)", fontsize=12, pad=12)
        axis.set_xlabel("Examples")
        tidy(axis)
        for i, value in enumerate(values):
            axis.text(value + total * 0.025, i, str(value), va="center", fontsize=11)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    fig.suptitle("Lowering the mushroom cutoff catches more poison and causes more false alarms",
                 fontsize=14, y=0.98)
    fig.text(0.02, 0.02, "800 development examples: 303 poisonous and 497 edible. "
             "Cautious cutoffs were chosen on separate calibration rows.", fontsize=9, color="#4b5563")
    fig.subplots_adjust(left=0.205, right=0.985, top=0.77, bottom=0.19, wspace=0.16)
    fig.savefig(REPORTS / "readme_mushroom_tradeoff.png", dpi=170, facecolor="white")
    plt.close(fig)


def bikes():
    with (REPORTS / "frozen_validation_comparison.csv").open(newline="") as stream:
        results = {row["name"]: row for row in csv.DictReader(stream)}
    names = ["control_lag7", "control_existing_xgboost_cpu_seed42",
             "blend_2023_history_and_forecasts", "blend_expanded_history_and_forecasts"]
    labels = ["Last week's count", "XGBoost / counts only",
              "CPU SVR + histogram boosting / forecasts", "GRU + CatBoost / forecasts"]
    values = [float(results[name]["mae"]) for name in names]
    fig, axis = plt.subplots(figsize=(10.6, 4.0))
    axis.barh(range(len(names)), values,
              color=["#9ca3af", "#397ca5", "#34866b", "#8b65a9"], height=0.6)
    axis.set_yticks(range(len(labels)), labels)
    axis.invert_yaxis()
    axis.set_xlim(0, max(values) * 1.13)
    axis.set_xlabel("Average absolute error (rides per day, lower is better)")
    axis.set_title("Citi Bike predictions: the same 92 days in July-September 2023", pad=15, fontsize=13)
    tidy(axis)
    for i, value in enumerate(values):
        axis.text(value + max(values) * 0.025, i, f"{value:,.0f}", va="center", fontsize=11)
    fig.text(0.02, 0.02, "Training ends in June. The GRU blend also uses 2022 history. "
             "These scores do not establish a dependable weather-model advantage.",
             fontsize=9, color="#4b5563")
    fig.subplots_adjust(left=0.35, right=0.985, top=0.82, bottom=0.2)
    fig.savefig(REPORTS / "readme_bike_errors.png", dpi=170, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    mushrooms()
    bikes()
    print("README figures regenerated from saved results. No models or tests were scored.")
