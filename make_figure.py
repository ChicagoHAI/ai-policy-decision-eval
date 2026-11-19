# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "pillow"]
# ///
"""Ranked column chart of evidence alignment, with vendor logos.

Usage: uv run make_figure.py   (reads results/graded_answers.csv, writes figures/*.png/.svg/.pdf)
Each chart is also written as *_bare: the bars and whiskers only, without title, subtitle or footnote.
"""
import csv
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch
from PIL import Image

HERE = Path(__file__).resolve().parent
CSV = HERE / "results" / "graded_answers.csv"
OUT = HERE / "figures"
LOGOS = OUT / "logos"

# alias -> (display name, logo file)
MODELS = {
    "fable-5.2": ("Claude\nFable 5.2", "claude-color"),
    "opus-5.5": ("Claude\nOpus 5.5", "claude-color"),
    "gpt-6-astra": ("GPT-6\nAstra", "openai"),
    "gemini-4-argon": ("Gemini 4\nArgon", "gemini-color"),
    "gemini-4-carbon": ("Gemini 4\nCarbon", "gemini-color"),
    "muse-wonder": ("Muse\nWonder", "meta-color"),
    "kimi-k3.1-pro": ("Kimi\nK3.1 Pro", "kimi-color"),
    "glm-5.3": ("GLM\n5.3", "zhipu-color"),
    "deepseek-v4-pro": ("DeepSeek\nV4 Pro", "deepseek-color"),
    "mimo-v2.6-pro": ("MiMo\nV2.6 Pro", "xiaomimimo"),
    "grok-4.7": ("Grok\n4.7", "grok"),
    "qwen3.8-max": ("Qwen3.8\nMax", "qwen-color"),
    "nemotron-3-ultra": ("Nemotron 3\nUltra", "nvidia-color"),
    "policymakers": ("Surveyed\npolicymakers*", None),
}

# palette (reference instance, light mode)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
SERIES, REFERENCE = "#2a78d6", "#898781"


def load(column):
    scores = defaultdict(list)
    for r in csv.DictReader(open(CSV)):
        if r[column] not in ("", None):
            scores[r["model"]].append(float(r[column]))
    return scores


def bootstrap_ci(xs, n=4000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def rounded_column(ax, x, h, w, color, radius_px=7):
    """Column with a rounded data-end and a square baseline (radius given in output pixels)."""
    bbox = ax.get_window_extent()
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    px_per_x = bbox.width / (x1 - x0)
    px_per_y = bbox.height / (y1 - y0)
    r = radius_px / px_per_x                       # radius in x data units
    ry = radius_px / px_per_y                      # same radius in y data units
    # one rounded shape that starts below the baseline; the axes clip its lower corners,
    # leaving a square baseline end with no seam
    ax.add_patch(FancyBboxPatch((x - w / 2, -2 * ry), w, h + 2 * ry,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=px_per_x / px_per_y, fc=color, ec="none", zorder=3))


def draw(column, title, subtitle, footnote, outname, include_baseline, bare=False):
    scores = load(column)
    rows = []
    for m, xs in scores.items():
        if m not in MODELS or (m == "policymakers" and not include_baseline):
            continue
        mean = 100 * sum(xs) / len(xs)
        lo, hi = bootstrap_ci(xs)
        rows.append((m, mean, 100 * lo, 100 * hi, len(xs)))
    rows.sort(key=lambda r: -r[1])

    plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    if bare:
        fig = plt.figure(figsize=(16, 4.9), dpi=200, facecolor=SURFACE)
        ax = fig.add_axes([0.045, 0.255, 0.94, 0.715], facecolor=SURFACE)
    else:
        fig = plt.figure(figsize=(16, 7.6), dpi=200, facecolor=SURFACE)
        ax = fig.add_axes([0.045, 0.30, 0.94, 0.52], facecolor=SURFACE)
    n = len(rows)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, 105)

    for y in (25, 50, 75, 100):
        ax.axhline(y, color=GRID, lw=0.8, zorder=0)
    ax.axhline(0, color=BASELINE, lw=1, zorder=4)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", colors=MUTED, labelsize=10, length=0, pad=6)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    w = 0.46
    for i, (m, mean, lo, hi, k) in enumerate(rows):
        color = REFERENCE if m == "policymakers" else SERIES
        rounded_column(ax, i, mean, w, color)
        ax.plot([i, i], [lo, hi], color=INK2, lw=1, alpha=0.55, zorder=5, solid_capstyle="butt")
        ax.plot([i - 0.07, i + 0.07], [lo, lo], color=INK2, lw=1, alpha=0.55, zorder=5)
        ax.plot([i - 0.07, i + 0.07], [hi, hi], color=INK2, lw=1, alpha=0.55, zorder=5)
        ax.text(i, max(mean, hi) + 1.8, f"{mean:.0f}", ha="center", va="bottom", fontsize=13,
                fontweight="bold", color=INK, zorder=6)

        name, logo = MODELS[m]
        if bare:
            name = name.rstrip("*")
        if logo:
            img = Image.open(LOGOS / f"{logo}.png").convert("RGBA")
            img.thumbnail((140, 140))
            ab = AnnotationBbox(OffsetImage(img, zoom=0.24), (i, 0), xybox=(0, -26),
                                xycoords="data", boxcoords="offset points", frameon=False)
            ax.add_artist(ab)
        ax.annotate(name, (i, 0), xytext=(0, -50), textcoords="offset points", ha="center",
                    va="top", fontsize=10.5, color=INK if logo else INK2, linespacing=1.15)

    if not bare:
        fig.text(0.045, 0.94, title, fontsize=19, fontweight="bold", color=INK, ha="left")
        fig.text(0.045, 0.895, subtitle, fontsize=11.5, color=INK2, ha="left")
        fig.text(0.045, 0.045, footnote, fontsize=8.6, color=MUTED, ha="left", va="bottom", linespacing=1.5)

    OUT.mkdir(exist_ok=True)
    if bare:
        outname += "_bare"
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{outname}.{ext}", facecolor=SURFACE)
    plt.close(fig)
    print(outname)
    for m, mean, lo, hi, k in rows:
        print(f"  {m:18} {mean:5.1f}  [{lo:5.1f}, {hi:5.1f}]  n={k}")


SOURCE = ("Questions and policymaker answers: Olken, Weiss and Hanna (2025), \"Social Protection: Policymaker "
          "Beliefs and Empirical Evidence\". One answer per model, November 2026.")


def main():
    for bare in (False, True):
        draw_both(bare)


def draw_both(bare):
    draw("final_evidence_score",
         "Which AI models' policy beliefs match the evidence on social protection?",
         "Evidence alignment score (0 to 100): agreement with the economics literature on 16 scored survey "
         "questions originally put to ministers and senior civil servants. Higher is better.",
         "Whiskers: 95% bootstrap interval over questions. Graded by Claude Opus 5.5 with partial credit for "
         "declined answers (0 where the literature is settled, 0.5 where it is a judgment call); questions with no "
         "single evidence-based answer are not scored.\n"
         "*Median/modal answer of 46 surveyed policymakers (excludes the country-specific payroll question). "
         + SOURCE,
         "evidence_ranking", include_baseline=True, bare=bare)
    draw("final_human_score",
         "Which AI models answer most like the surveyed policymakers?",
         "Policymaker agreement score (0 to 100): closeness to the 46 policymakers' aggregate answer on all 19 "
         "survey items. Higher means closer to policymakers, not closer to the evidence.",
         "Whiskers: 95% bootstrap interval over questions. Graded by Claude Opus 5.5 against the modal choice, "
         "median scale answer, mean ranking order or median number from the survey; declined answers score 0.\n"
         + SOURCE,
         "policymaker_agreement", include_baseline=False, bare=bare)


if __name__ == "__main__":
    main()
