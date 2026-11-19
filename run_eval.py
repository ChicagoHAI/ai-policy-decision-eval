# /// script
# requires-python = ">=3.10"
# dependencies = ["anthropic", "openai", "pyyaml"]
# ///
"""Ask AI models the social protection survey questions, then grade answers with LLM judges.

Usage (uv installs the dependencies automatically):
  uv run run_eval.py all                                   # answerers + judges from config.yml
  uv run run_eval.py answer --models opus-5.5 kimi-k3.1-pro --samples 3
  uv run run_eval.py judge  --judges opus-5.5
  uv run run_eval.py report
  uv run run_eval.py list                                  # show configured models

Models are referred to by their alias in config.yml (use --config for another file).

Outputs (all append-only; reruns skip work already done):
  results/answers.jsonl   one row per (model, question, sample) with the raw response
  results/grades.jsonl    one row per (answer, judge)
  results/summary.csv     mean scores per model (written by `report`)
"""

import argparse
import concurrent.futures as cf
import csv
import json
import re
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE / "social_protection_questions.jsonl"
OUT = HERE / "results"
ANSWERS = OUT / "answers.jsonl"
GRADES = OUT / "grades.jsonl"

_lock = threading.Lock()
_clients = {}


# ---------------------------------------------------------------- config & providers

CONFIG = {}


def load_config(path):
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for alias, m in cfg.get("models", {}).items():
        if m.get("provider") not in cfg.get("providers", {}):
            sys.exit(f"config: model '{alias}' uses unknown provider '{m.get('provider')}'")
    CONFIG.clear()
    CONFIG.update(cfg)


def model_cfg(alias):
    try:
        m = CONFIG["models"][alias]
    except KeyError:
        sys.exit(f"Unknown model alias '{alias}'. Known: {', '.join(CONFIG['models'])}")
    return m, CONFIG["providers"][m["provider"]]


def client_for(provider_name):
    import os
    with _lock:
        if provider_name not in _clients:
            p = CONFIG["providers"][provider_name]
            env = p.get("api_key_env", "")
            key = p.get("api_key") or os.environ.get(env)
            if not key and re.match(r"^(sk-|xai-|nvapi-)", env):
                print(f"WARNING: providers.{provider_name}.api_key_env holds a key, not an env var "
                      f"name; using it directly. Prefer exporting it as an env var.", file=sys.stderr)
                key = env
            if not key:
                raise RuntimeError(f"env var {env} is not set (provider {provider_name})")
            if p["type"] == "anthropic":
                import anthropic
                _clients[provider_name] = anthropic.Anthropic(api_key=key, max_retries=5)
            else:
                import openai
                _clients[provider_name] = openai.OpenAI(api_key=key, base_url=p.get("base_url"),
                                                        max_retries=5)
        return _clients[provider_name]


def call_model(alias, prompt, system=None, schema=None, effort=None, fallbacks=False, followup=None):
    """Return dict(text, served_model, stop_reason, usage). Raises on API errors.

    `effort` overrides the model's configured effort when given.
    `followup` is an optional list of {"role", "content"} turns sent after the prompt
    (used to continue the conversation, e.g. a retry reminder)."""
    turns = [{"role": "user", "content": prompt}] + list(followup or [])
    m, p = model_cfg(alias)
    client = client_for(m["provider"])
    name = m["model"]
    effort = effort or m.get("effort")
    max_tokens = m.get("max_tokens", CONFIG.get("defaults", {}).get("max_tokens", 16000))
    extra = dict(m.get("extra") or {})

    if p["type"] == "anthropic":
        kwargs = dict(model=name, max_tokens=max_tokens,
                      messages=turns, **extra)
        if system:
            kwargs["system"] = system
        output_config = {}
        if effort:
            output_config["effort"] = effort
        if schema:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        if output_config:
            kwargs["output_config"] = output_config
        # streaming avoids HTTP timeouts on long thinking turns (e.g. Fable 5.2)
        if fallbacks:
            with client.beta.messages.stream(betas=["server-side-fallback-2026-07-01"],
                                             fallbacks="default", **kwargs) as s:
                resp = s.get_final_message()
        else:
            with client.messages.stream(**kwargs) as s:
                resp = s.get_final_message()
        text = "".join(b.text for b in resp.content if b.type == "text")
        return dict(text=text, served_model=resp.model, stop_reason=resp.stop_reason,
                    usage=dict(input_tokens=resp.usage.input_tokens,
                               output_tokens=resp.usage.output_tokens))

    messages = ([{"role": "system", "content": system}] if system else []) + turns
    kwargs = dict(model=name, messages=messages, **extra)
    if effort:
        if p.get("style") == "openrouter":
            body = dict(kwargs.pop("extra_body", None) or {})
            body["reasoning"] = {**body.get("reasoning", {}), "effort": effort}
            kwargs["extra_body"] = body
        else:
            kwargs["reasoning_effort"] = effort
    kwargs.setdefault("max_tokens", max_tokens)
    if schema:
        mode = m.get("json_mode", "object")
        if mode == "schema":
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "grade", "schema": schema, "strict": True}}
        elif mode == "object":
            kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    return dict(text=choice.message.content or "", served_model=resp.model,
                stop_reason=choice.finish_reason,
                usage=dict(input_tokens=getattr(resp.usage, "prompt_tokens", None),
                           output_tokens=getattr(resp.usage, "completion_tokens", None)))


# ---------------------------------------------------------------- io helpers

def load_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path, row):
    with _lock:
        with open(path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_json(text):
    """Pull the first JSON object out of a model response (handles ```json fences)."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [m.group(1)] if m else []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_parallel(jobs, fn, workers):
    done = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fn, j) for j in jobs]
        for fut in cf.as_completed(futures):
            done += 1
            try:
                msg = fut.result()
            except Exception as e:  # keep going; failures are logged, not fatal
                msg = f"ERROR {type(e).__name__}: {e}"
            print(f"[{done}/{len(jobs)}] {msg}", flush=True)


# ---------------------------------------------------------------- stage 1: answer

RETRY_REMINDER = ("Your reply did not contain the required JSON answer. Even if the answer depends on "
                  "context, you must commit to the single answer you consider best on balance, as the "
                  "surveyed policymakers had to. Reply with only the JSON object in the format requested "
                  "above, with no other text.")


def latest_answers():
    """Latest answer row per (model, question, sample); retries append an updated row."""
    out = {}
    for r in load_jsonl(ANSWERS):
        out[(r["model"], r["question_id"], r["sample"])] = r
    return out


def answer_version(a):
    return a.get("retry_timestamp") or a["timestamp"]


def cmd_answer(args):
    OUT.mkdir(exist_ok=True)
    questions = select_questions(args)
    existing = latest_answers()
    jobs, n_saved = [], 0
    for model in args.models:
        for q in questions:
            for s in range(args.samples):
                old = existing.get((model, q["id"], s))
                if old and old.get("error") is None:
                    if old.get("parsed") is None and not old.get("retried"):
                        jobs.append(("retry", model, q, s, old))   # saved but unparsed: retry only
                    else:
                        n_saved += 1
                    continue
                prompt = q["prompt"]
                if "{country}" in prompt:
                    prompt = prompt.replace("{country}", args.country)
                jobs.append(("new", model, q, s, prompt))
    n_retry = sum(1 for j in jobs if j[0] == "retry")
    print(f"{len(jobs) - n_retry} new answer calls, {n_retry} retries of unparsed answers "
          f"({n_saved} already saved)")

    def retry(row):
        """Send the reminder in the same conversation; update row in place."""
        model = row["model"]
        row.update(retried=True, first_text=row["text"], first_usage=row.get("usage"),
                   first_stop_reason=row.get("stop_reason"), retry_timestamp=now())
        try:
            r = call_model(model, row["prompt"], system=row.get("system"), effort=row.get("effort"),
                           fallbacks=args.fallbacks,
                           followup=[{"role": "assistant", "content": row["text"] or "(empty)"},
                                     {"role": "user", "content": RETRY_REMINDER}])
            row.update(r, parsed=extract_json(r["text"]), retry_error=None)
        except Exception as e:
            row["retry_error"] = f"{type(e).__name__}: {e}"

    def work(job):
        kind, model, q, s, payload = job
        t0 = time.time()
        if kind == "retry":
            row = dict(payload)
        else:
            row = dict(model=model, question_id=q["id"], sample=s, prompt=payload,
                       system=args.system, effort=args.effort, timestamp=now(), retried=False)
            try:
                r = call_model(model, payload, system=args.system, effort=args.effort,
                               fallbacks=args.fallbacks)
                row.update(r, parsed=extract_json(r["text"]), error=None)
            except Exception as e:
                row.update(text=None, parsed=None, error=f"{type(e).__name__}: {e}")
        if row.get("error") is None and row["parsed"] is None and not row.get("retried"):
            retry(row)
        row["latency_s"] = round(row.get("latency_s", 0) + time.time() - t0, 2)
        append_jsonl(ANSWERS, row)
        if row.get("error"):
            status = row["error"]
        elif row.get("retry_error"):
            status = f"UNPARSED, retry failed: {row['retry_error']}"
        elif row["parsed"] is not None:
            status = "parsed after retry" if row.get("retried") else "parsed"
        else:
            status = "UNPARSED after retry"
        return f"{model} {q['id']} s{s}: {status}"

    run_parallel(jobs, work, args.workers)


# ---------------------------------------------------------------- stage 2: judge

JUDGE_SYSTEM = """You are grading an AI model's answer to a survey question that was originally \
posed to senior policymakers (ministers and civil servants) in low- and middle-income countries. \
A chapter by Olken, Weiss and Hanna (2025) compared the policymakers' answers with the empirical \
economics literature. You are given the question, the answer options, the policymakers' aggregate \
answer, a summary of what the literature says (with a gold answer when one exists), and the AI's \
answer. Grade only what the AI committed to in its answer; do not reward hedging that avoids \
committing. Be consistent across answers."""

JUDGE_TEMPLATE = """## Question ({qid}, answer type: {answer_type})
{question}

## Options
{options}

## Policymakers' answer (human survey)
{human}

## What the literature says
{literature}

## The authors' verdict on whether policymakers matched the evidence
{agreement}

## AI model's answer
{answer}

## How to grade
1. answer_status: one of
   - "committed": the AI chose an option / gave a full ranking / gave a number, even if it adds caveats.
   - "declined": the AI deliberately refused to choose because it thinks the answer depends on \
context or there is no single answer (e.g. "it depends", "cannot rank"), possibly with a discussion.
   - "invalid": empty, off-topic, truncated, or otherwise gives no usable position.
   valid_answer is true only for "committed".
   answer_extracted: restate the committed answer in a short phrase (for rankings, the order; \
for numbers, the number). For declined answers, summarize any lean it expressed in a few words.
   For "declined" or "invalid", set evidence_score and human_agreement_score to null (they are \
scored separately by the evaluation code) and evidence_gap_direction to "not_applicable".
2. evidence_score (0 to 1): how well the AI's answer agrees with the literature.
   - If a gold answer is given, score against it. Rankings: credit for putting the literature-favored \
options near the top/bottom as indicated. Likert: 1 if the answer is within the gold set, 0.5 if \
one step away in the right direction, 0 otherwise. Numeric: 1 if within the gold range or within \
5 points of it, decreasing linearly to 0 at 40+ points away.
   - If gold is null (the literature gives no single answer), set evidence_score to null.
3. human_agreement_score (0 to 1): how closely the AI matches the policymakers' aggregate answer \
(modal/median choice, mean ranking order, or mean/median number).
4. evidence_gap_direction: if the policymakers deviate from the evidence on this question, did the \
AI side with "evidence", "policymakers", "neither", or is it "not_applicable" (no gap or no gold)?
5. rationale: 1 to 3 sentences explaining the scores.

Return only a JSON object with exactly these keys: answer_status ("committed" | "declined" | \
"invalid"), valid_answer (boolean), answer_extracted (string), \
evidence_score (number or null), human_agreement_score (number or null), evidence_gap_direction \
("evidence" | "policymakers" | "neither" | "not_applicable"), rationale (string)."""

JUDGE_VERSION = 2  # bump when the judge prompt/schema changes; older grades are redone

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer_status": {"type": "string", "enum": ["committed", "declined", "invalid"]},
        "valid_answer": {"type": "boolean"},
        "answer_extracted": {"type": "string"},
        "evidence_score": {"type": ["number", "null"]},
        "human_agreement_score": {"type": ["number", "null"]},
        "evidence_gap_direction": {"type": "string",
                                   "enum": ["evidence", "policymakers", "neither", "not_applicable"]},
        "rationale": {"type": "string"},
    },
    "required": ["answer_status", "valid_answer", "answer_extracted", "evidence_score", "human_agreement_score",
                 "evidence_gap_direction", "rationale"],
    "additionalProperties": False,
}


def judge_prompt(q, ans):
    opts = q.get("options")
    return JUDGE_TEMPLATE.format(
        qid=q["id"], answer_type=q["answer_type"],
        question=q["question"] + (f" (sub-item: {q['sub_item']})" if q.get("sub_item") else ""),
        options="\n".join(f"- {o}" for o in opts) if opts else "(free numeric answer, percent)",
        human=json.dumps(q["human_answer"], ensure_ascii=False, indent=1),
        literature=json.dumps(q["literature_answer"], ensure_ascii=False, indent=1),
        agreement=q.get("agreement", "unknown"),
        answer=format_answer(ans),
    )


def format_answer(ans):
    final = ans["text"] if ans.get("text") else "(no answer)"
    if not ans.get("retried"):
        return final
    return ("[First reply, which did not contain the requested JSON answer]\n"
            f"{ans.get('first_text') or '(empty)'}\n\n"
            "[The AI was then reminded that it must commit to one answer. Final reply]\n"
            f"{final}")


def cmd_judge(args):
    OUT.mkdir(exist_ok=True)
    qmap = {q["id"]: q for q in load_jsonl(QUESTIONS)}
    answers = [a for a in latest_answers().values() if a.get("error") is None]
    if args.models:
        answers = [a for a in answers if a["model"] in args.models]
    done = {}
    for g in load_jsonl(GRADES):
        if g.get("error") is None:
            if g.get("judge_version") == JUDGE_VERSION:
                done[(g["model"], g["question_id"], g["sample"], g["judge"])] = g.get("answer_version")

    def is_done(a, j):
        key = (a["model"], a["question_id"], a["sample"], j)
        if key not in done:
            return False
        return done[key] == answer_version(a)

    jobs = [(a, j) for a in answers for j in args.judges if not is_done(a, j)]
    print(f"{len(jobs)} judge calls to make ({len(done)} already saved)")

    def work(job):
        ans, judge = job
        q = qmap[ans["question_id"]]
        row = dict(model=ans["model"], question_id=q["id"], sample=ans["sample"],
                   judge=judge, judge_version=JUDGE_VERSION, answer_version=answer_version(ans),
                   timestamp=now())
        try:
            r = call_model(judge, judge_prompt(q, ans), system=JUDGE_SYSTEM,
                           schema=JUDGE_SCHEMA, effort=args.judge_effort)
            grade = extract_json(r["text"])
            missing = [k for k in JUDGE_SCHEMA["required"] if not grade or k not in grade]
            if missing:
                raise ValueError(f"judge output missing {missing}: {r['text'][:200]!r}")
            row.update(grade=grade, judge_raw=r["text"], usage=r["usage"], error=None)
        except Exception as e:
            row.update(grade=None, error=f"{type(e).__name__}: {e}")
        append_jsonl(GRADES, row)
        if row["error"]:
            return f"{judge} on {ans['model']} {q['id']}: {row['error']}"
        g = row["grade"]
        return (f"{judge} on {ans['model']} {q['id']} s{ans['sample']}: {g['answer_status']} "
                f"evidence={g['evidence_score']} human={g['human_agreement_score']}")

    run_parallel(jobs, work, args.workers)


# ---------------------------------------------------------------- stage 3: report

def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


DEFAULT_SCORING = {
    "declined_credit": {"clear": 0.0, "soft": 0.5, "none": None},
    "invalid_credit": 0.0,
    "declined_human_agreement": 0.0,
    "invalid_human_agreement": 0.0,
}


def final_scores(grade, q, scoring):
    """Apply the scoring policy. Returns (status, evidence, human)."""
    status = grade.get("answer_status") or ("committed" if grade.get("valid_answer") else "invalid")
    strength = q.get("gold_strength", "clear")
    if status == "committed":
        ev = grade.get("evidence_score")
        return status, (None if strength == "none" else ev), grade.get("human_agreement_score")
    if status == "declined":
        return status, scoring["declined_credit"].get(strength), scoring["declined_human_agreement"]
    return status, (None if strength == "none" else scoring["invalid_credit"]), \
        scoring["invalid_human_agreement"]


def cmd_report(args):
    grades = [g for g in load_jsonl(GRADES) if g.get("error") is None]
    if not grades:
        sys.exit("No grades yet. Run `answer` and `judge` first.")
    # keep only the latest grade per (model, question, sample, judge)
    latest = {}
    for g in grades:
        latest[(g["model"], g["question_id"], g["sample"], g["judge"])] = g
    grades = list(latest.values())
    stale = sum(1 for g in grades if g.get("judge_version") != JUDGE_VERSION)
    if stale:
        print(f"WARNING: {stale} grades come from an older judge version; run `judge` to redo them.\n")

    scoring = {**DEFAULT_SCORING, **(CONFIG.get("scoring") or {})}
    scoring["declined_credit"] = {**DEFAULT_SCORING["declined_credit"],
                                  **(scoring.get("declined_credit") or {})}
    qmap = {q["id"]: q for q in load_jsonl(QUESTIONS)}
    answers = latest_answers()
    by_model = defaultdict(list)
    by_mq = defaultdict(list)
    for g in grades:
        q = qmap[g["question_id"]]
        g["_status"], g["_ev"], g["_hu"] = final_scores(g["grade"], q, scoring)
        g["_strength"] = q.get("gold_strength", "clear")
        by_model[g["model"]].append(g)
        by_mq[(g["model"], g["question_id"])].append(g)

    def rate(gs, status):
        return mean([1.0 if g["_status"] == status else 0.0 for g in gs])

    rows = []
    for model, gs in sorted(by_model.items()):
        gap = [g["grade"]["evidence_gap_direction"] for g in gs if g["_status"] == "committed"
               and g["grade"]["evidence_gap_direction"] in ("evidence", "policymakers", "neither")]
        rows.append(dict(
            model=model, n_grades=len(gs),
            committed_rate=rate(gs, "committed"),
            declined_rate=rate(gs, "declined"),
            invalid_rate=rate(gs, "invalid"),
            retry_rate=mean([1.0 if answers.get((g["model"], g["question_id"], g["sample"]), {}).get("retried")
                             else 0.0 for g in gs]),
            evidence_score=mean([g["_ev"] for g in gs]),
            evidence_committed_only=mean([g["_ev"] for g in gs if g["_status"] == "committed"]),
            human_agreement=mean([g["_hu"] for g in gs]),
            declined_on_open_questions=sum(1 for g in gs if g["_status"] == "declined" and g["_strength"] == "none"),
            sides_with_evidence_when_gap=(round(gap.count("evidence") / len(gap), 3) if gap else None),
            n_gap_items=len(gap),
        ))
    OUT.mkdir(exist_ok=True)
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(OUT / "per_question.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "question_id", "gold_strength", "n", "committed", "declined", "invalid",
                    "evidence_score", "human_agreement", "answers"])
        for (model, qid), gs in sorted(by_mq.items()):
            w.writerow([model, qid, gs[0]["_strength"], len(gs),
                        sum(g["_status"] == "committed" for g in gs),
                        sum(g["_status"] == "declined" for g in gs),
                        sum(g["_status"] == "invalid" for g in gs),
                        mean([g["_ev"] for g in gs]), mean([g["_hu"] for g in gs]),
                        " | ".join(sorted({g["grade"]["answer_extracted"] for g in gs}))])

    with open(OUT / "graded_answers.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "question_id", "sample", "judge", "gold_strength", "answer_status",
                    "answer_extracted", "retried", "judge_evidence_score", "final_evidence_score",
                    "judge_human_score", "final_human_score", "evidence_gap_direction",
                    "rationale", "answer_text"])
        for g in sorted(grades, key=lambda g: (g["model"], g["question_id"], g["sample"], g["judge"])):
            gr = g["grade"]
            a = answers.get((g["model"], g["question_id"], g["sample"]), {})
            w.writerow([g["model"], g["question_id"], g["sample"], g["judge"], g["_strength"],
                        g["_status"], gr.get("answer_extracted"), bool(a.get("retried")),
                        gr.get("evidence_score"), g["_ev"], gr.get("human_agreement_score"), g["_hu"],
                        gr.get("evidence_gap_direction"), gr.get("rationale"), a.get("text")])

    fmt = lambda v: "-" if v is None else f"{v:.2f}"
    hdr = (f"{'model':18} {'n':>4} {'commit':>6} {'declin':>6} {'invalid':>7} {'retry':>6} "
           f"{'evidence':>8} {'ev|commit':>9} {'human':>6} {'sides_w_evid':>13}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['model']:18} {r['n_grades']:>4} {fmt(r['committed_rate']):>6} {fmt(r['declined_rate']):>6} "
              f"{fmt(r['invalid_rate']):>7} {fmt(r['retry_rate']):>6} {fmt(r['evidence_score']):>8} "
              f"{fmt(r['evidence_committed_only']):>9} {fmt(r['human_agreement']):>6} "
              f"{fmt(r['sides_with_evidence_when_gap']):>9} ({r['n_gap_items']})")
    dc = scoring["declined_credit"]
    print(f"\nDeclined answers get evidence credit clear={dc['clear']}, soft={dc['soft']}, "
          f"none={dc['none']} (not scored); invalid answers get {scoring['invalid_credit']}.")
    print(f"Wrote {OUT/'summary.csv'}, {OUT/'per_question.csv'} and {OUT/'graded_answers.csv'}")


# ---------------------------------------------------------------- cli

def select_questions(args):
    qs = load_jsonl(QUESTIONS)
    if args.questions:
        qs = [q for q in qs if q["id"] in args.questions]
    if args.core_only:
        qs = [q for q in qs if q.get("is_core_ten", True)]
    return qs


def cmd_list(args):
    for alias, m in CONFIG["models"].items():
        tags = [t for t, lst in (("answerer", CONFIG.get("answerers", [])), ("judge", CONFIG.get("judges", [])))
                if alias in lst]
        print(f"{alias:18} {m['provider']:11} {m['model']:36} {','.join(tags)}")


def main():
    global OUT, ANSWERS, GRADES
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(HERE / "config.yml"))
    p.add_argument("--out", default=str(OUT), help="results directory")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_answer_args(sp):
        sp.add_argument("--models", nargs="+", help="model aliases (default: answerers in config)")
        sp.add_argument("--samples", type=int, help="answers per model per question")
        sp.add_argument("--questions", nargs="+", help="only these question ids")
        sp.add_argument("--core-only", action="store_true", help="skip the D2 side question")
        sp.add_argument("--country", default="a typical lower-middle-income country",
                        help="fills {country} in question G1")
        sp.add_argument("--system", default=None, help="optional system prompt for answerers")
        sp.add_argument("--effort", default=None, help="override every answerer's effort")
        sp.add_argument("--fallbacks", action="store_true",
                        help="enable Claude server-side refusal fallbacks (answer may come from another model)")

    def add_judge_args(sp):
        sp.add_argument("--judges", nargs="+", help="judge aliases (default: judges in config)")
        sp.add_argument("--judge-effort", default=None)

    a = sub.add_parser("answer"); add_answer_args(a)
    j = sub.add_parser("judge"); add_judge_args(j)
    j.add_argument("--models", nargs="+", help="only grade answers from these models")
    sub.add_parser("report")
    sub.add_parser("list")
    al = sub.add_parser("all"); add_answer_args(al); add_judge_args(al)
    for sp in (a, j, al):
        sp.add_argument("--workers", type=int)

    args = p.parse_args()
    OUT = Path(args.out)
    ANSWERS, GRADES = OUT / "answers.jsonl", OUT / "grades.jsonl"
    load_config(args.config)
    d = CONFIG.get("defaults", {})
    if getattr(args, "workers", 0) is None:
        args.workers = d.get("workers", 8)
    if getattr(args, "samples", 0) is None:
        args.samples = d.get("samples", 1)
    if args.cmd in ("answer", "all") and not args.models:
        args.models = CONFIG.get("answerers", [])
    if args.cmd in ("judge", "all") and not args.judges:
        args.judges = CONFIG.get("judges", [])
    to_check = list(getattr(args, "judges", None) or [])
    if args.cmd in ("answer", "all"):  # for `judge`, --models only filters saved answers
        to_check += args.models or []
    for alias in to_check:
        model_cfg(alias)  # fail fast on typos

    {"answer": cmd_answer, "judge": cmd_judge, "report": cmd_report, "list": cmd_list}.get(
        args.cmd, lambda a: (cmd_answer(a), cmd_judge(a), cmd_report(a)))(args)


if __name__ == "__main__":
    main()
