# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Build the Hugging Face dataset folder from the questions file and results/.

Usage: uv run build_hf_dataset.py OUT_DIR
Then:  hf upload ChicagoHAI/ai-policy-decision-eval OUT_DIR --repo-type dataset

Nested fields whose types vary across rows (gold answers, policymaker answer distributions,
parsed model answers) are stored as JSON strings so the dataset viewer can load every table.
The original JSONL files are copied unchanged into raw/.
"""

import csv
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def dumps(v):
    return None if v is None else json.dumps(v, ensure_ascii=False)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def csv_rows(path):
    """Read a CSV; columns where every non-empty value is numeric become floats, others stay strings."""
    rows = list(csv.DictReader(open(path, newline="")))
    for col in rows[0].keys() if rows else []:
        vals = [r[col] for r in rows if r[col] != ""]
        try:
            [float(v) for v in vals]
            numeric = bool(vals)
        except ValueError:
            numeric = False
        for r in rows:
            v = r[col]
            r[col] = None if v == "" else (float(v) if numeric else v)
    return rows


def to_float(v):
    return None if v is None else float(v)


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    questions = []
    for q in read_jsonl(HERE / "social_protection_questions.jsonl"):
        lit, hum = q["literature_answer"], q["human_answer"]
        questions.append({
            "id": q["id"],
            "section": q["section"],
            "topic": q["topic"],
            "question": q["question"],
            "answer_type": q["answer_type"],
            "options": q["options"] or [],
            "prompt": q["prompt"],
            "gold_strength": q["gold_strength"],
            "gold_strength_note": q["gold_strength_note"],
            "literature_summary": lit.get("summary"),
            "literature_gold": dumps(lit.get("gold")),
            "literature_key_citations": lit.get("key_citations") or [],
            "policymaker_answer": dumps(hum),
            "policymakers_match_evidence": q["agreement"],
            "is_core_ten": q["is_core_ten"],
            "pages": q["pages"],
            "source": q["source"],
        })
    write_jsonl(out / "data/questions.jsonl", questions)

    answers = []
    for a in read_jsonl(HERE / "results/answers.jsonl"):
        a = dict(a)
        a["parsed"] = dumps(a.get("parsed"))
        a["usage"] = dumps(a.get("usage"))
        a["latency_s"] = to_float(a.get("latency_s"))
        a["effort"] = None if a.get("effort") is None else str(a["effort"])
        a["error"] = None if a.get("error") is None else str(a["error"])
        answers.append(a)
    write_jsonl(out / "data/answers.jsonl", answers)

    grades = []
    for g in read_jsonl(HERE / "results/grades.jsonl"):
        gr = g.get("grade") or {}
        grades.append({
            "model": g["model"],
            "question_id": g["question_id"],
            "sample": g["sample"],
            "judge": g["judge"],
            "judge_version": g.get("judge_version"),
            "timestamp": g.get("timestamp"),
            "answer_status": gr.get("answer_status"),
            "valid_answer": gr.get("valid_answer"),
            "answer_extracted": None if gr.get("answer_extracted") is None else str(gr["answer_extracted"]),
            "evidence_score": to_float(gr.get("evidence_score")),
            "human_agreement_score": to_float(gr.get("human_agreement_score")),
            "evidence_gap_direction": gr.get("evidence_gap_direction"),
            "rationale": gr.get("rationale"),
            "judge_raw": g.get("judge_raw"),
            "error": None if g.get("error") is None else str(g["error"]),
        })
    write_jsonl(out / "data/grades.jsonl", grades)

    # CSV tables are converted to JSONL: with mixed formats in one repo, `datasets` reads every file as JSON
    for name in ["graded_answers", "per_question", "summary"]:
        write_jsonl(out / f"data/{name}.jsonl", csv_rows(HERE / f"results/{name}.csv"))

    raw = out / "raw"
    raw.mkdir(exist_ok=True)
    shutil.copy(HERE / "social_protection_questions.jsonl", raw)
    shutil.copy(HERE / "results/answers.jsonl", raw)
    shutil.copy(HERE / "results/grades.jsonl", raw)
    shutil.copy(HERE / "figures/evidence_ranking.png", out)
    print(f"wrote {out}: {len(questions)} questions, {len(answers)} answers, {len(grades)} grades")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
