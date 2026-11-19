# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml"]
# ///
"""Write LaTeX tables for the paper from results/graded_answers.csv.

Usage: uv run make_tables.py   (writes paper/tables/*.tex)
"""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
OUT = HERE / "paper" / "tables"
BASELINE = "policymakers"

NAMES = {
    "fable-5.2": "Claude Fable 5.2", "opus-5.5": "Claude Opus 5.5", "gpt-6-astra": "GPT-6 Astra",
    "gemini-4-argon": "Gemini 4 Argon", "gemini-4-carbon": "Gemini 4 Carbon", "muse-wonder": "Muse Wonder",
    "kimi-k3.1-pro": "Kimi K3.1 Pro", "glm-5.3": "GLM 5.3", "deepseek-v4-pro": "DeepSeek V4 Pro",
    "mimo-v2.6-pro": "MiMo V2.6 Pro", "grok-4.7": "Grok 4.7", "qwen3.8-max": "Qwen3.8 Max",
    "nemotron-3-ultra": "Nemotron 3 Ultra",
}
DEVELOPER = {
    "fable-5.2": "Anthropic", "opus-5.5": "Anthropic", "gpt-6-astra": "OpenAI", "gemini-4-argon": "Google",
    "gemini-4-carbon": "Google", "muse-wonder": "Meta", "kimi-k3.1-pro": "Moonshot AI", "glm-5.3": "Z.ai",
    "deepseek-v4-pro": "DeepSeek", "mimo-v2.6-pro": "Xiaomi", "grok-4.7": "xAI", "qwen3.8-max": "Alibaba",
    "nemotron-3-ultra": "NVIDIA",
}
LABELS = {
    "A1": "Best form of transfer (rank)", "B_consumption": "Cash transfers: consumption",
    "B_food_security": "Cash transfers: food security", "B_school_attendance": "Cash transfers: school attendance",
    "B_health": "Cash transfers: health", "B_high_school_graduation": "Cash transfers: high school graduation",
    "B_temptation_goods": "Cash transfers: alcohol, tobacco, drugs", "B_labor_supply": "Cash transfers: labor supply",
    "B_fertility": "Cash transfers: fertility", "C1": "Metric of program success (rank)",
    "D1": "Targeted or universal", "D2": "Trend in global extreme poverty",
    "E1": "Best targeting method (rank)", "F1": "Conditional or unconditional",
    "G1": "Payroll-tax insurance coverage (\\%)", "H1": "Enrollment in self-paid health insurance (\\%)",
    "I1": "Effect of subsidized health insurance", "J1": "Effect of social spending on votes",
    "J2": "Most popular program with voters (rank)",
}

f = lambda x: float(x) if x not in ("", None) else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def ci(xs, n=4000, seed=0):
    rng = random.Random(seed)
    m = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n)]


def fmt(x, d=0, scale=100):
    return "--" if x is None else f"{x * scale:.{d}f}"


def main():
    rows = list(csv.DictReader(open(HERE / "results" / "graded_answers.csv")))
    qs = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "social_protection_questions.jsonl")}
    cfg = yaml.safe_load(open(HERE / "config.yml"))
    bym, byq = defaultdict(list), defaultdict(list)
    for r in rows:
        bym[r["model"]].append(r)
        byq[r["question_id"]].append(r)
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- Table 1: main results
    stats = []
    for m, rs in bym.items():
        ev = [f(r["final_evidence_score"]) for r in rs if f(r["final_evidence_score"]) is not None]
        evc = [f(r["final_evidence_score"]) for r in rs
               if r["answer_status"] == "committed" and f(r["final_evidence_score"]) is not None]
        gap = [r["evidence_gap_direction"] for r in rs if r["answer_status"] == "committed"
               and r["evidence_gap_direction"] in ("evidence", "policymakers", "neither")]
        lo, hi = ci(ev)
        stats.append(dict(m=m, ev=mean(ev), lo=lo, hi=hi, evc=mean(evc),
                          hu=mean([f(r["final_human_score"]) for r in rs]),
                          gap=(gap.count("evidence") / len(gap)) if gap else None, ngap=len(gap),
                          dec=sum(r["answer_status"] == "declined" for r in rs), n=len(rs)))
    stats.sort(key=lambda s: -s["ev"])
    lines, rank = [], 0
    for s in stats:
        if s["m"] == BASELINE:
            lines.append(r"\addlinespace[2pt]" "\n"
                         f"\\textit{{Policymakers (median answer)}} & & \\textit{{{fmt(s['ev'])}}} & "
                         f"\\textit{{[{fmt(s['lo'])}, {fmt(s['hi'])}]}} & & & & \\\\\n\\addlinespace[2pt]")
            continue
        rank += 1
        lines.append(f"{NAMES[s['m']]} & {rank} & {fmt(s['ev'])} & [{fmt(s['lo'])}, {fmt(s['hi'])}] & "
                     f"{fmt(s['evc'])} & {fmt(s['hu'], 2, 1)} & {fmt(s['gap'], 2, 1)} ({s['ngap']}) & "
                     f"{s['dec']} \\\\")
    (OUT / "main_results.tex").write_text("\n".join(lines) + "\n")

    # ---- Table 2: by question
    lines = []
    for qid, q in qs.items():
        rs = [r for r in byq[qid] if r["model"] != BASELINE]
        base = [r for r in byq[qid] if r["model"] == BASELINE]
        ev = mean([f(r["final_evidence_score"]) for r in rs])
        pev = f(base[0]["final_evidence_score"]) if base else None
        hu = mean([f(r["final_human_score"]) for r in rs])
        dec = sum(r["answer_status"] == "declined" for r in rs)
        lines.append(f"{LABELS[qid]} & {q['gold_strength']} & {fmt(ev)} & "
                     f"{fmt(pev) if base else 'n.a.'} & {fmt(hu, 2, 1)} & {dec} \\\\")
    (OUT / "by_question.tex").write_text("\n".join(lines) + "\n")

    # ---- Appendix table: models
    lines = []
    for alias in cfg["answerers"]:
        mid = cfg["models"][alias]["model"].replace("_", r"\_")
        eff = cfg["models"][alias].get("effort", "default")
        lines.append(f"{NAMES[alias]} & {DEVELOPER[alias]} & \\texttt{{{mid}}} & {eff} \\\\")
    (OUT / "models.tex").write_text("\n".join(lines) + "\n")

    # ---- numbers used in the text
    s_by = {s["m"]: s for s in stats}
    models = [s for s in stats if s["m"] != BASELINE]
    above = sum(1 for s in models if s["ev"] > s_by[BASELINE]["ev"])
    mrows = [r for r in rows if r["model"] != BASELINE]
    scored = [f(r["final_evidence_score"]) for r in mrows if f(r["final_evidence_score"]) is not None]
    evs = sorted(s["ev"] for s in models)
    # item-level comparison with the policymaker baseline
    better = tie = worse = 0
    for qid in qs:
        base = [f(r["final_evidence_score"]) for r in byq[qid] if r["model"] == BASELINE]
        mv = mean([f(r["final_evidence_score"]) for r in byq[qid] if r["model"] != BASELINE])
        if not base or base[0] is None or mv is None:
            continue
        d = mv - base[0]
        better, tie, worse = better + (d >= 0.05), tie + (abs(d) < 0.05), worse + (d <= -0.05)

    def corr(ms):
        xs, ys = [s["ev"] for s in ms], [s["hu"] for s in ms]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
        return num / den

    macros = {
        "NModels": len(models),
        "NAbove": above,
        "BaselineScore": fmt(s_by[BASELINE]["ev"]),
        "TopScore": fmt(stats[0]["ev"]),
        "BottomScore": fmt(stats[-1]["ev"]),
        "MeanScore": fmt(mean([s["ev"] for s in models])),
        "MedianScore": fmt(evs[len(evs) // 2]),
        "NAnswers": len(mrows),
        "NCommitted": sum(r["answer_status"] == "committed" for r in mrows),
        "NDeclined": sum(r["answer_status"] == "declined" for r in mrows),
        "NInvalid": sum(r["answer_status"] == "invalid" for r in mrows),
        "NScored": len(scored),
        "FullCreditPct": fmt(sum(x >= 0.999 for x in scored) / len(scored)),
        "ZeroCreditPct": fmt(sum(x == 0 for x in scored) / len(scored)),
        "NBetter": better, "NTie": tie, "NWorse": worse,
        "CorrAll": f"${corr(models):.2f}$",
        "CorrExDecliner": f"${corr([s for s in models if s['dec'] == 0]):.2f}$",
    }
    (OUT / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
    print("wrote", ", ".join(p.name for p in OUT.iterdir()), macros)


if __name__ == "__main__":
    main()
