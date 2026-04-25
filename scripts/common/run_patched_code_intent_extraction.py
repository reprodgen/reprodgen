from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from reprodbench.pipeline.judge_patched_code_intent import PatchedCodeIntentJudge
from reprodbench.pipeline.patched_code_intent import (
    PatchedCodeIntentExtractor,
    PatchedCodeIntentRefiner,
)
from reprodbench.utils.logger import log_section, log_step, setup_logger

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

load_dotenv()
logger = setup_logger("reprodbench.patched_code_intent")

MAX_GENERATOR_RETRIES = 3
MAX_JUDGE_RETRIES = 3
MAX_CYCLES = 2  # extractor ↔ judge refinement cycles

MODEL_DATASETS = {
   
    "gpt-oss:20b": "path_to_guidance",
}

JUDGE_MODEL_NAME = "qwen3:8b"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

RESULTS_DIR = PROJECT_ROOT / "results" / "github" / "patched_ci"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GEN_PROMPT_DIR = (
    PROJECT_ROOT
    / "src"
    / "reprodbench"
    / "llm"
    / "prompts"
    / "thought_generation"
    / "patched"
)

JUDGE_PROMPT_DIR = (
    PROJECT_ROOT / "src" / "reprodbench" / "llm" / "prompts" / "judge_llm" / "patched"
)


def now_ts():
    return datetime.now().isoformat(timespec="seconds")


def safe_token(tokens, key):
    if not tokens:
        return None
    return tokens.get(key)


def run_for_model(MODEL_NAME: str, DATA_PATH: str):
    dataset = pd.read_csv(DATA_PATH)

    RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_PATH = RESULTS_DIR / f"patched_code_intent_{MODEL_NAME}_{RUN_TS}.csv"

    log_section(
        logger,
        "CONFIG",
        {
            "MODEL_NAME": MODEL_NAME,
            "JUDGE_MODEL_NAME": JUDGE_MODEL_NAME,
            "DATA_PATH": str(DATA_PATH),
            "MAX_GENERATOR_RETRIES": MAX_GENERATOR_RETRIES,
            "MAX_JUDGE_RETRIES": MAX_JUDGE_RETRIES,
            "MAX_CYCLES": MAX_CYCLES,
            "RESULTS_PATH": str(RESULTS_PATH),
        },
    )

    extractor = PatchedCodeIntentExtractor(
        prompt_dir=GEN_PROMPT_DIR,
        provider="ollama",
        model=MODEL_NAME,
        temperature=1.0,
    )

    refiner = PatchedCodeIntentRefiner(
        prompt_dir=GEN_PROMPT_DIR,
        provider="ollama",
        model=MODEL_NAME,
        temperature=1.0,
    )

    judge = PatchedCodeIntentJudge(
        prompt_dir=JUDGE_PROMPT_DIR,
        provider="ollama",
        model=JUDGE_MODEL_NAME,
        temperature=1.0,
    )

    write_header = True

    def write_row(row):
        nonlocal write_header
        pd.DataFrame([row]).to_csv(
            RESULTS_PATH,
            mode="a",
            index=False,
            header=write_header,
        )
        write_header = False

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------
    for idx, row in dataset.iterrows():
        print("\n" + "=" * 100)
        print(f"[{idx + 1}/{len(dataset)}] Model: {MODEL_NAME}")
        print(f"Question ID: {row.question_id}")
        print("Link:", row.question_link)
        print("=" * 100)

        question = row.question_title + "\n" + row.question_body
        answer = row.accepted_answer_body

        patched_intent = None

        # ---------------------------
        # EXTRACTOR
        # ---------------------------
        gen_latency = None
        gen_tokens = None

        # ---------------------------
        # FINAL (console-only) state
        # ---------------------------
        final_label = None
        final_rationale = None
        final_judge_latency = None
        final_judge_tokens = None

        for gen_attempt in range(1, MAX_GENERATOR_RETRIES + 1):
            try:
                log_step(logger, f"[{MODEL_NAME}] EXTRACTOR ATTEMPT {gen_attempt}")

                artifact = extractor.extract(
                    question=question,
                    answer=answer,
                )
                log_step(
                    logger, f"[{MODEL_NAME}] Artifact: {artifact.patched_code_intent}"
                )
                patched_intent = artifact.patched_code_intent
                gen_latency = artifact.latency
                gen_tokens = artifact.token_usage
                break

            except Exception as e:
                log_step(logger, f"[{MODEL_NAME}] EXTRACTOR ERROR: {e}")

        if patched_intent is None:
            write_row(
                {
                    "model_name": MODEL_NAME,
                    "question_id": row.question_id,
                    "question_link": row.question_link,
                    "cycle": None,
                    "attempt": None,
                    "stage": "extractor_failed",
                    "patched_code_intent": None,
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

        # ---------------------------
        # JUDGE–REFINE CYCLES
        # ---------------------------
        for cycle in range(1, MAX_CYCLES + 1):
            log_step(logger, f"[{MODEL_NAME}] CYCLE {cycle}/{MAX_CYCLES}")

            judge_result = None

            for judge_attempt in range(1, MAX_JUDGE_RETRIES + 1):
                try:
                    judge_result = judge.judge(
                        question=question,
                        answer=answer,
                        patched_code_intent=patched_intent,
                    )
                    log_step(
                        logger, f"[{MODEL_NAME}] judge_result: {judge_result.label}"
                    )
                    break
                except Exception as e:
                    log_step(logger, f"[{MODEL_NAME}] JUDGE ERROR: {e}")

            if judge_result is None:
                break

            final_label = judge_result.label
            final_rationale = judge_result.rationale
            final_judge_latency = judge_result.latency
            final_judge_tokens = judge_result.token_usage

            write_row(
                {
                    "model_name": MODEL_NAME,
                    "question_id": row.question_id,
                    "question_link": row.question_link,
                    "cycle": cycle,
                    "attempt": judge_attempt,
                    "stage": "judge",
                    "patched_code_intent": patched_intent,
                    "generator_latency": gen_latency,
                    "generator_input_tokens": safe_token(gen_tokens, "input_tokens"),
                    "generator_output_tokens": safe_token(gen_tokens, "output_tokens"),
                    "generator_total_tokens": safe_token(gen_tokens, "total_tokens"),
                    "judge_label": final_label,
                    "judge_rationale": final_rationale,
                    "judge_latency": final_judge_latency,
                    "judge_input_tokens": safe_token(
                        final_judge_tokens, "input_tokens"
                    ),
                    "judge_output_tokens": safe_token(
                        final_judge_tokens, "output_tokens"
                    ),
                    "judge_total_tokens": safe_token(
                        final_judge_tokens, "total_tokens"
                    ),
                    "timestamp": now_ts(),
                }
            )

            if final_label == "correct":
                break

            refined_artifact = None

            for refine_attempt in range(1, MAX_GENERATOR_RETRIES + 1):
                try:
                    log_step(
                        logger,
                        f"[{MODEL_NAME}] REFINER ATTEMPT {refine_attempt}/{MAX_GENERATOR_RETRIES}",
                    )

                    refined_artifact = refiner.refine(
                        question=question,
                        answer=answer,
                        patched_code_intent=patched_intent,
                        judge_label=final_label,
                        judge_rationale=final_rationale,
                    )
                    break

                except ValueError as e:
                    log_step(
                        logger,
                        f"[{MODEL_NAME}] REFINER STRUCTURED OUTPUT ERROR (attempt {refine_attempt}): {e}",
                    )

                except Exception as e:
                    log_step(
                        logger,
                        f"[{MODEL_NAME}] REFINER ERROR (attempt {refine_attempt}): {e}",
                    )

            if refined_artifact is None:
                write_row(
                    {
                        "model_name": MODEL_NAME,
                        "question_id": row.question_id,
                        "question_link": row.question_link,
                        "cycle": cycle,
                        "attempt": refine_attempt,
                        "stage": "refiner_failed",
                        "patched_code_intent": patched_intent,
                        "generator_latency": gen_latency,
                        "generator_input_tokens": safe_token(
                            gen_tokens, "input_tokens"
                        ),
                        "generator_output_tokens": safe_token(
                            gen_tokens, "output_tokens"
                        ),
                        "generator_total_tokens": safe_token(
                            gen_tokens, "total_tokens"
                        ),
                        "judge_label": final_label,
                        "judge_rationale": final_rationale,
                        "judge_latency": final_judge_latency,
                        "judge_input_tokens": safe_token(
                            final_judge_tokens, "input_tokens"
                        ),
                        "judge_output_tokens": safe_token(
                            final_judge_tokens, "output_tokens"
                        ),
                        "judge_total_tokens": safe_token(
                            final_judge_tokens, "total_tokens"
                        ),
                        "timestamp": now_ts(),
                    }
                )
                break

            patched_intent = refined_artifact.patched_code_intent

        # ---------------------------
        # CONSOLE SUMMARY
        # ---------------------------
        print("\n" + "-" * 80)
        print("FINAL PATCHED CODE INTENT")
        print("-" * 80)
        print(f"Model       : {MODEL_NAME}")
        print(f"Question ID : {row.question_id}")
        print(f"Final Label : {final_label}")

        print("\n[GENERATOR METRICS]")
        print(f"Latency        : {gen_latency}")
        print(f"Input Tokens   : {safe_token(gen_tokens, 'input_tokens')}")
        print(f"Output Tokens  : {safe_token(gen_tokens, 'output_tokens')}")
        print(f"Total Tokens   : {safe_token(gen_tokens, 'total_tokens')}")

        print("\n[JUDGE METRICS]")
        print(f"Latency        : {final_judge_latency}")
        print(f"Input Tokens   : {safe_token(final_judge_tokens, 'input_tokens')}")
        print(f"Output Tokens  : {safe_token(final_judge_tokens, 'output_tokens')}")
        print(f"Total Tokens   : {safe_token(final_judge_tokens, 'total_tokens')}")

        print("\nPatched Code Intent:")
        print(patched_intent)

        print("\nJudge Rationale:")
        print(final_rationale)

        print("-" * 80)

    log_step(logger, f"[{MODEL_NAME}] RESULTS WRITTEN → {RESULTS_PATH}")


for MODEL_NAME, DATA_PATH in MODEL_DATASETS.items():
    run_for_model(MODEL_NAME, DATA_PATH)
