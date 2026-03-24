from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATASETS = ["CaseReportBench", "PHEE", "DiscourseEE", "MACCROBAT"]
QG_Models = ["qwen3-4b", "qwen3-8b", "gpt-oss-120b", "gpt-5.2-high", "gemini-3.1-pro-high"]
PD_Models = ["qwen3-4b", "qwen3-8b", "gpt-oss-120b", "gpt-5-mini-medium", "gemini-3.1-pro-high"]

OUTPUTS_SC_DIR = "/dartfs/rc/home/j/f006f3j/lab/omar/LoQA/Outputs/sc"
SETUP = "loqa"
METRIC = "complex-match-f1"
SAVE_PATH = "/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA/Code/xDev/loqa_scores_palette.png"

# Display names for figure text
QG_DISPLAY_NAMES = {
    "qwen3-4b": "Qwen3-4B",
    "qwen3-8b": "Qwen3-8B",
    "gpt-oss-120b": "GPT-Oss-120B",
    "gpt-5.2-high": "GPT-5.2",
    "gemini-3.1-pro-high": "Gemini-3.1-Pro",
}
PD_DISPLAY_NAMES = {
    "qwen3-4b": "Qwen3-4B",
    "qwen3-8b": "Qwen3-8B",
    "gpt-oss-120b": "GPT-Oss-120B",
    "gpt-5-mini-medium": "GPT-5-mini",
    "gemini-3.1-pro-high": "Gemini-3.1-Pro",
}
#X_AXIS_LABEL = "Question Generation Model"
Y_AXIS_LABEL = "F1 Score" if METRIC == "complex-match-f1" else "Relaxed Match F1 (%)"


def load_scores(dataset: str, setup: str, qg_model: str, pd_model: str):
    base_dir = Path(OUTPUTS_SC_DIR) / dataset
    filename = f"{setup}-{qg_model}-zs-v0-{dataset}-dev-{pd_model}-zs-v0.json"
    path = base_dir / filename
    if not path.exists():
        return None, None

    with path.open("r") as f:
        data = json.load(f).get("overall", {})

    relaxed_f1 = data.get("relaxed-match-f1")
    complex_f1 = data.get("complex-match-f1")
    return relaxed_f1, complex_f1


datasets_dfs = {}
for dataset in DATASETS:
    rows = []
    for qg_model in QG_Models:
        row = {"QG": qg_model}
        for pd_model in PD_Models:
            relaxed, complex_ = load_scores(dataset, SETUP, qg_model, pd_model)
            row[(pd_model, "relaxed-match-f1")] = relaxed
            row[(pd_model, "complex-match-f1")] = complex_
        rows.append(row)

    df = pd.DataFrame(rows).set_index("QG")
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["PD_model", "metric"])
    datasets_dfs[dataset] = df

PD_COLOR_MAP = {
    "qwen3-4b": "#0072B2",
    "qwen3-8b": "#E69F00",
    "gpt-oss-120b": "#009E73",
    "gpt-5-mini-medium": "#D55E00",
    "gemini-3.1-pro-high": "#7A7A7A"
}
PD_HATCH_MAP = {
    "qwen3-4b": "",
    "qwen3-8b": "//",
    "gpt-oss-120b": "\\\\",
    "gpt-5-mini-medium": "-",
    "gemini-3.1-pro-high": "..",
}

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "DejaVu Sans"

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
axes = axes.flatten()

handles_legend = None
for ax, dataset in zip(axes, DATASETS):
    df = datasets_dfs[dataset]
    df_plot = df.xs(METRIC, axis=1, level=1).astype(float)

    x = np.arange(len(QG_Models))
    width = 0.8 / len(PD_Models)

    for i, pd_model in enumerate(PD_Models):
        offset = (i - len(PD_Models) / 2 + 0.5) * width
        vals = df_plot[pd_model].values

        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=PD_DISPLAY_NAMES.get(pd_model, pd_model),
            color=PD_COLOR_MAP[pd_model],
            edgecolor="black",
            linewidth=0.7,
            hatch=PD_HATCH_MAP[pd_model],
            alpha=0.95,
        )

        for bar, v in zip(bars, vals):
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.2,
                    f"{float(v):.1f}",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    if handles_legend is None:
        handles_legend, labels_legend = ax.get_legend_handles_labels()

    ax.set_xticks(x)
    ax.set_xticklabels([QG_DISPLAY_NAMES.get(m, m) for m in QG_Models], rotation=25, ha="right")
    # ax.set_xlabel(X_AXIS_LABEL)
    ax.set_ylabel(Y_AXIS_LABEL)
    ax.set_title(dataset, fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.legend(
    handles_legend,
    labels_legend,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=len(PD_Models),
    frameon=True,
    fontsize=9,
    columnspacing=1.2,
)
plt.tight_layout(rect=[0, 0.02, 1, 0.93])
fig.savefig(SAVE_PATH, bbox_inches="tight")
print(SAVE_PATH)
