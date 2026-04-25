from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from reprodbench.pipeline.buggy_code_intent import (
    BuggyCodeIntentExtractor,
    BuggyCodeIntentRefiner,
)
from reprodbench.pipeline.judge_buggy_code_intent import BuggyCodeIntentJudge
from reprodbench.utils.logger import log_section, log_step, setup_logger

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

load_dotenv()
logger = setup_logger("reprodbench.buggy_code_intent")

MAX_GENERATOR_RETRIES = 3
MAX_JUDGE_RETRIES = 3
MAX_REFINER_RETRIES = 3
MAX_CYCLES = 2  # generator ↔ judge refinement cycles

GEN_MODELS = [
    # "phi4:latest",
    # "gemma3:27b",
    # "codellama:7b",
    # "qwen2.5-coder:32b",
    # "deepseek-coder-v2:16b",
    "gpt-oss:20b"
]

JUDGE_MODEL_NAME = "qwen3:8b"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "run_github_issues_sampled.csv"

RESULTS_DIR = PROJECT_ROOT / "results" / "github" / "buggy_ci"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GEN_PROMPT_DIR = (
    PROJECT_ROOT
    / "src"
    / "reprodbench"
    / "llm"
    / "prompts"
    / "thought_generation"
    / "buggy"
)

JUDGE_PROMPT_DIR = (
    PROJECT_ROOT / "src" / "reprodbench" / "llm" / "prompts" / "judge_llm" / "buggy"
)

dataset = pd.read_csv(DATA_PATH)


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def now_ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_get(d: dict | None, key: str):
    if not isinstance(d, dict):
        return None
    return d.get(key)


def flatten_token_usage(prefix: str, usage: dict | None) -> dict:
    input_tokens = _safe_get(usage, "prompt_tokens")
    if input_tokens is None:
        input_tokens = _safe_get(usage, "input_tokens")

    output_tokens = _safe_get(usage, "completion_tokens")
    if output_tokens is None:
        output_tokens = _safe_get(usage, "output_tokens")

    total_tokens = _safe_get(usage, "total_tokens")
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)

    return {
        f"{prefix}_input_tokens": input_tokens,
        f"{prefix}_output_tokens": output_tokens,
        f"{prefix}_total_tokens": total_tokens,
    }


def flatten_metrics(prefix: str, latency: float | None, usage: dict | None) -> dict:
    out = {f"{prefix}_latency": latency}
    out.update(flatten_token_usage(prefix, usage))
    return out


for MODEL_NAME in GEN_MODELS:
    safe_model = MODEL_NAME.replace(":", "_")
    safe_judge = JUDGE_MODEL_NAME.replace(":", "_")
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_path = (
        RESULTS_DIR / f"buggy_code_intent_{safe_model}_{safe_judge}_{run_ts}.csv"
    )

    print(f"\n\n{'=' * 100}")
    print(f"RUNNING MODEL: {MODEL_NAME}")
    print(f"JUDGE MODEL  : {JUDGE_MODEL_NAME}")
    print(f"RESULTS PATH : {results_path}")
    print(f"{'=' * 100}\n")

    log_section(
        logger,
        "CONFIG",
        {
            "MODEL_NAME": MODEL_NAME,
            "JUDGE_MODEL_NAME": JUDGE_MODEL_NAME,
            "MAX_GENERATOR_RETRIES": MAX_GENERATOR_RETRIES,
            "MAX_JUDGE_RETRIES": MAX_JUDGE_RETRIES,
            "MAX_REFINER_RETRIES": MAX_REFINER_RETRIES,
            "MAX_CYCLES": MAX_CYCLES,
            "RESULTS_PATH": str(results_path),
        },
    )

    write_header_state = {"value": True}

    def write_row(row: dict):
        pd.DataFrame([row]).to_csv(
            results_path,
            mode="a",
            index=False,
            header=write_header_state["value"],
        )
        write_header_state["value"] = False

    extractor = BuggyCodeIntentExtractor(
        prompt_dir=GEN_PROMPT_DIR,
        provider="ollama",
        model=MODEL_NAME,
        temperature=1.0,
    )

    refiner = BuggyCodeIntentRefiner(
        prompt_dir=GEN_PROMPT_DIR,
        provider="ollama",
        model=MODEL_NAME,
        temperature=1.0,
    )

    judge = BuggyCodeIntentJudge(
        prompt_dir=JUDGE_PROMPT_DIR,
        provider="ollama",
        model=JUDGE_MODEL_NAME,
        temperature=1.0,
    )

    for idx, row in dataset.iterrows():
        print("\n" + "=" * 100)
        print(f"[{idx + 1}/{len(dataset)}] Question ID: {row.question_id}")
        print("Link:", row.question_link)
        print("=" * 100)

        question = row.question_title + "\n" + row.question_body

        # --------------------------------------------------------------
        # GENERATOR
        # --------------------------------------------------------------
        gen_artifact = None
        gen_err = None

        for gen_attempt in range(1, MAX_GENERATOR_RETRIES + 1):
            try:
                log_step(logger, f"[{row.question_id}] GENERATOR ATTEMPT {gen_attempt}")
                gen_artifact = extractor.extract(question_text=question)
                gen_err = None
                break
            except Exception as e:
                gen_err = str(e)
                log_step(logger, f"[{row.question_id}] GENERATOR ERROR: {gen_err}")

        gen_intent = None
        gen_latency = None
        gen_usage = None

        if gen_artifact is not None:
            try:
                if hasattr(gen_artifact, "buggy_intent"):
                    gen_intent = getattr(gen_artifact, "buggy_intent")
                    gen_latency = getattr(gen_artifact, "latency", None)
                    gen_usage = getattr(gen_artifact, "token_usage", None)
                else:
                    gen_intent = str(gen_artifact)
                    gen_latency = getattr(
                        getattr(extractor, "metrics", None), "latency", None
                    )
                    gen_usage = getattr(
                        getattr(extractor, "metrics", None), "token_usage", None
                    )
            except Exception as e:
                gen_err = f"generator_output_parse_error: {e}"
                gen_intent = None

        if gen_intent is None:
            write_row(
                {
                    "model_name": MODEL_NAME,
                    "judge_model_name": JUDGE_MODEL_NAME,
                    "question_id": row.question_id,
                    "question_link": row.question_link,
                    "cycle": None,
                    "attempt": gen_attempt,
                    "stage": "generator_failed",
                    "buggy_code_intent": None,
                    "judge_label": None,
                    "judge_rationale": None,
                    "error": gen_err,
                    "timestamp": now_ts(),
                    **flatten_metrics("generator", gen_latency, gen_usage),
                    **flatten_metrics("judge", None, None),
                }
            )
            continue

        # --------------------------------------------------------------
        # JUDGE–REFINE CYCLES
        # --------------------------------------------------------------
        current_intent = gen_intent
        last_judge_label = None
        last_judge_rationale = None

        for cycle in range(1, MAX_CYCLES + 1):
            log_step(logger, f"[{row.question_id}] CYCLE {cycle}/{MAX_CYCLES}")

            judge_result = None
            judge_err = None

            for judge_attempt in range(1, MAX_JUDGE_RETRIES + 1):
                try:
                    judge_result = judge.judge(
                        question=question,
                        buggy_code_intent=current_intent,
                    )
                    judge_err = None
                    break
                except Exception as e:
                    judge_err = str(e)
                    log_step(
                        logger,
                        f"[{row.question_id}] JUDGE ERROR (attempt {judge_attempt}): {judge_err}",
                    )

            if judge_result is None:
                write_row(
                    {
                        "model_name": MODEL_NAME,
                        "judge_model_name": JUDGE_MODEL_NAME,
                        "question_id": row.question_id,
                        "question_link": row.question_link,
                        "cycle": cycle,
                        "attempt": judge_attempt,
                        "stage": "judge_failed",
                        "buggy_code_intent": current_intent,
                        "judge_label": None,
                        "judge_rationale": None,
                        "error": judge_err,
                        "timestamp": now_ts(),
                        **flatten_metrics("generator", gen_latency, gen_usage),
                        **flatten_metrics("judge", None, None),
                    }
                )
                break

            j_label = getattr(judge_result, "label", None)
            j_rationale = getattr(judge_result, "rationale", None)
            j_latency = getattr(judge_result, "latency", None)
            j_usage = getattr(judge_result, "token_usage", None)

            last_judge_label = j_label
            last_judge_rationale = j_rationale

            write_row(
                {
                    "model_name": MODEL_NAME,
                    "judge_model_name": JUDGE_MODEL_NAME,
                    "question_id": row.question_id,
                    "question_link": row.question_link,
                    "cycle": cycle,
                    "attempt": judge_attempt,
                    "stage": "judge",
                    "buggy_code_intent": current_intent,
                    "judge_label": j_label,
                    "judge_rationale": j_rationale,
                    "error": None,
                    "timestamp": now_ts(),
                    **flatten_metrics("generator", gen_latency, gen_usage),
                    **flatten_metrics("judge", j_latency, j_usage),
                }
            )

            if j_label == "correct":
                break

            # ----------------------------------------------------------
            # REFINER
            # ----------------------------------------------------------
            refined = None
            refine_err = None

            for refine_attempt in range(1, MAX_REFINER_RETRIES + 1):
                try:
                    refined = refiner.refine(
                        question=question,
                        buggy_intent=current_intent,
                        judge_label=j_label,
                        judge_rationale=j_rationale,
                    )
                    refine_err = None
                    break
                except Exception as e:
                    refine_err = str(e)
                    log_step(
                        logger,
                        f"[{row.question_id}] REFINER ERROR (attempt {refine_attempt}): {refine_err}",
                    )
                    write_row(
                        {
                            "model_name": MODEL_NAME,
                            "judge_model_name": JUDGE_MODEL_NAME,
                            "question_id": row.question_id,
                            "question_link": row.question_link,
                            "cycle": cycle,
                            "attempt": refine_attempt,
                            "stage": "refiner_failed",
                            "buggy_code_intent": current_intent,
                            "judge_label": j_label,
                            "judge_rationale": j_rationale,
                            "error": refine_err,
                            "timestamp": now_ts(),
                            **flatten_metrics("generator", gen_latency, gen_usage),
                            **flatten_metrics("judge", j_latency, j_usage),
                        }
                    )

            if refined is None:
                break

            try:
                current_intent = (
                    getattr(refined, "buggy_intent", None)
                    or getattr(refined, "intent", None)
                    or str(refined)
                )
            except Exception as e:
                write_row(
                    {
                        "model_name": MODEL_NAME,
                        "judge_model_name": JUDGE_MODEL_NAME,
                        "question_id": row.question_id,
                        "question_link": row.question_link,
                        "cycle": cycle,
                        "attempt": None,
                        "stage": "refiner_output_parse_failed",
                        "buggy_code_intent": None,
                        "judge_label": j_label,
                        "judge_rationale": j_rationale,
                        "error": str(e),
                        "timestamp": now_ts(),
                        **flatten_metrics("generator", gen_latency, gen_usage),
                        **flatten_metrics("judge", j_latency, j_usage),
                    }
                )
                break

        # --------------------------------------------------------------
        # CONSOLE SUMMARY
        # --------------------------------------------------------------
        try:
            print("\n" + "-" * 80)
            print("FINAL BUGGY CODE INTENT RESULT")
            print("-" * 80)
            print(f"Model       : {MODEL_NAME}")
            print(f"Question ID : {row.question_id}")
            print(f"Final Label : {last_judge_label}")
            print("\nBuggy Code Intent:")
            print(current_intent)
            print("-" * 80)
        except Exception as e:
            log_step(
                logger,
                f"[{row.question_id}] FINAL PRINT ERROR: {e}",
            )
            write_row(
                {
                    "model_name": MODEL_NAME,
                    "judge_model_name": JUDGE_MODEL_NAME,
                    "question_id": row.question_id,
                    "question_link": row.question_link,
                    "cycle": None,
                    "attempt": None,
                    "stage": "final_print_failed",
                    "buggy_code_intent": None,
                    "judge_label": last_judge_label,
                    "judge_rationale": last_judge_rationale,
                    "error": str(e),
                    "timestamp": now_ts(),
                    **flatten_metrics("generator", gen_latency, gen_usage),
                    **flatten_metrics("judge", None, None),
                }
            )

    log_step(logger, f"RESULTS WRITTEN → {results_path}")
