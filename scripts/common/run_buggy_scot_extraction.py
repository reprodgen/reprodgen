from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from reprodbench.pipeline.buggy_scot import BuggyScotExtractor, BuggyScotRefiner
from reprodbench.pipeline.judge_buggy_scot import BuggyScotJudge
from reprodbench.utils.logger import log_section, log_step, setup_logger

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

load_dotenv()
logger = setup_logger("reprodbench.buggy_scot")

MAX_GENERATOR_RETRIES = 3
MAX_JUDGE_RETRIES = 3
MAX_CYCLES = 2  # extractor ↔ judge refinement cycles

GEN_MODELS = [
    # "phi4:latest",
    # "gemma3:27b",
    # "deepseek-coder-v2:16b",
    # "qwen2.5-coder:32b",
    # "codellama:7b",
    "gpt-oss:20b"
]

JUDGE_MODEL_NAME = "qwen3:8b"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "run_github_issues_sampled.csv"


RESULTS_DIR = PROJECT_ROOT / "results" / "github" / "buggy_scot"
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


# ----------------------------------------------------------------------
# RUN PER MODEL
# ----------------------------------------------------------------------


def run_for_model(MODEL_NAME: str):
    RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_PATH = RESULTS_DIR / f"buggy_scot_{MODEL_NAME}_{RUN_TS}.csv"

    log_section(
        logger,
        "CONFIG",
        {
            "MODEL_NAME": MODEL_NAME,
            "JUDGE_MODEL_NAME": JUDGE_MODEL_NAME,
            "MAX_GENERATOR_RETRIES": MAX_GENERATOR_RETRIES,
            "MAX_JUDGE_RETRIES": MAX_JUDGE_RETRIES,
            "MAX_CYCLES": MAX_CYCLES,
            "RESULTS_PATH": str(RESULTS_PATH),
        },
    )

    # ----------------------------------------------------------------------
    # INIT
    # ----------------------------------------------------------------------

    extractor = BuggyScotExtractor(
        prompt_dir=GEN_PROMPT_DIR,
        provider="ollama",
        model=MODEL_NAME,
        temperature=1.0,
    )

    refiner = BuggyScotRefiner(
        prompt_dir=GEN_PROMPT_DIR,
        provider="ollama",
        model=MODEL_NAME,
        temperature=1.0,
    )

    judge = BuggyScotJudge(
        prompt_dir=JUDGE_PROMPT_DIR,
        provider="ollama",
        model=JUDGE_MODEL_NAME,
        temperature=1.0,
    )

    write_header = True

    def write_row(row: dict):
        nonlocal write_header
        pd.DataFrame([row]).to_csv(
            RESULTS_PATH,
            mode="a",
            index=False,
            header=write_header,
        )
        write_header = False

    # ----------------------------------------------------------------------
    # MAIN LOOP
    # ----------------------------------------------------------------------

    for idx, row in dataset.iterrows():
        print("\n" + "=" * 100)
        print(f"[{idx + 1}/{len(dataset)}] Question ID: {row.question_id}")
        print("Link:", row.question_link)
        print("=" * 100)

        question = row.question_title + "\n" + row.question_body
        buggy_scot = None

        # ---------------------------
        # Metrics (per question)
        # ---------------------------
        gen_latency = None
        gen_tokens = None

        # ---------------------------
        # Final (console-only) state
        # ---------------------------
        final_scot = None
        final_label = None
        final_rationale = None
        final_cycle = None

        # --------------------------------------------------------------
        # EXTRACTOR
        # --------------------------------------------------------------

        for gen_attempt in range(1, MAX_GENERATOR_RETRIES + 1):
            try:
                log_step(logger, f"EXTRACTOR ATTEMPT {gen_attempt}")

                extractor.metrics.reset()
                artifact = extractor.extract(question=question)

                buggy_scot = artifact.buggy_scot
                gen_latency = extractor.metrics.latency
                gen_tokens = extractor.metrics.token_usage
                break

            except Exception as e:
                log_step(logger, f"EXTRACTOR ERROR: {e}")

        if buggy_scot is None:
            write_row(
                {
                    "question_id": row.question_id,
                    "question_link": row.question_link,
                    "cycle": None,
                    "attempt": None,
                    "stage": "extractor_failed",
                    "buggy_scot": None,
                    "generator_latency": gen_latency,
                    "generator_input_tokens": None,
                    "generator_output_tokens": None,
                    "generator_total_tokens": None,
                    "judge_label": None,
                    "judge_rationale": None,
                    "judge_latency": None,
                    "judge_input_tokens": None,
                    "judge_output_tokens": None,
                    "judge_total_tokens": None,
                    "timestamp": now_ts(),
                }
            )
            continue

        # --------------------------------------------------------------
        # JUDGE–REFINE CYCLES
        # --------------------------------------------------------------

        for cycle in range(1, MAX_CYCLES + 1):
            log_step(logger, f"CYCLE {cycle}/{MAX_CYCLES}")

            judge_result = None

            for judge_attempt in range(1, MAX_JUDGE_RETRIES + 1):
                try:
                    judge.metrics.reset()
                    judge_result = judge.judge(
                        question=question,
                        buggy_scot=buggy_scot,
                    )
                    break
                except Exception as e:
                    log_step(logger, f"JUDGE ERROR: {e}")

            if judge_result is None:
                break

            # ---------------------------
            # Track final (console only)
            # ---------------------------
            final_scot = buggy_scot
            final_label = judge_result.label
            final_rationale = judge_result.rationale
            final_cycle = cycle

            write_row(
                {
                    "question_id": row.question_id,
                    "question_link": row.question_link,
                    "cycle": cycle,
                    "attempt": judge_attempt,
                    "stage": "judge",
                    "buggy_scot": buggy_scot,
                    # Generator metrics
                    "generator_latency": gen_latency,
                    "generator_input_tokens": (
                        gen_tokens.get("input_tokens") if gen_tokens else None
                    ),
                    "generator_output_tokens": (
                        gen_tokens.get("output_tokens") if gen_tokens else None
                    ),
                    "generator_total_tokens": (
                        gen_tokens.get("total_tokens") if gen_tokens else None
                    ),
                    # Judge outputs
                    "judge_label": judge_result.label,
                    "judge_rationale": judge_result.rationale,
                    # Judge metrics
                    "judge_latency": judge.metrics.latency,
                    "judge_input_tokens": (
                        judge.metrics.token_usage.get("input_tokens")
                        if judge.metrics.token_usage
                        else None
                    ),
                    "judge_output_tokens": (
                        judge.metrics.token_usage.get("output_tokens")
                        if judge.metrics.token_usage
                        else None
                    ),
                    "judge_total_tokens": (
                        judge.metrics.token_usage.get("total_tokens")
                        if judge.metrics.token_usage
                        else None
                    ),
                    "timestamp": now_ts(),
                }
            )

            if judge_result.label == "correct":
                break  # stop cycles entirely

            # --------------------------------------------------
            # REFINER (only if NOT correct)
            # --------------------------------------------------

            refined_artifact = None

            for refine_attempt in range(1, MAX_GENERATOR_RETRIES + 1):
                try:
                    log_step(
                        logger,
                        f"REFINER ATTEMPT {refine_attempt}/{MAX_GENERATOR_RETRIES}",
                    )

                    refined_artifact = refiner.refine(
                        question=question,
                        buggy_scot=buggy_scot,
                        judge_label=judge_result.label,
                        judge_rationale=judge_result.rationale,
                    )
                    break

                except ValueError as e:  # <-- correct exception
                    log_step(
                        logger,
                        f"REFINER STRUCTURED OUTPUT ERROR (attempt {refine_attempt}): {e}",
                    )

            if refined_artifact is None:
                log_step(
                    logger,
                    "REFINER FAILED AFTER RETRIES — STOPPING CYCLE",
                )
                break  # stop cycles

            buggy_scot = refined_artifact.buggy_scot

        # --------------------------------------------------------------
        # FINAL CONSOLE SUMMARY
        # --------------------------------------------------------------

        print("\n" + "-" * 80)
        print("FINAL BUGGY SCOT RESULT")
        print("-" * 80)
        print(f"Question ID : {row.question_id}")
        print(f"Final Cycle : {final_cycle}")
        print(f"Final Label : {final_label}")

        if final_scot:
            print("\nBuggy SCoT:")
            print(final_scot)

        if final_rationale:
            print("\nJudge Rationale:")
            print(final_rationale)

        print("-" * 80 + "\n")

    log_step(logger, f"RESULTS WRITTEN → {RESULTS_PATH}")


# ----------------------------------------------------------------------
# RUN ALL MODELS
# ----------------------------------------------------------------------

for MODEL_NAME in GEN_MODELS:
    run_for_model(MODEL_NAME)
