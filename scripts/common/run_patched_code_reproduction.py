import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from reprodbench.ablation.builder import build_patched_semantic_context
from reprodbench.ablation.mode import AblationMode
from reprodbench.executor.sandbox_client import ExecutionResult, SandboxExecutorClient
from reprodbench.pipeline.judge_patched_code import PatchedCodeJudge
from reprodbench.pipeline.patched_code import (
    PatchedCodeGenerator,
    PatchedCodeRefiner,
    StructuredOutputError,
)
from reprodbench.utils.logger import log_section, log_step, setup_logger
from reprodbench.utils.text import preview

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

load_dotenv()
logger = setup_logger("reprodbench.patched_pipeline")

MAX_GENERATOR_RETRIES = 3
MAX_REFINER_RETRIES = 3
MAX_EXEC_RETRIES = 3
MAX_JUDGE_RETRIES = 3
MAX_CYCLES = 3
TIMEOUT_SECONDS = 120

JUDGE_MODEL_NAME = "qwen3:8b"

MAX_DOCKER_ERROR_CHARS = 800
MAX_TRACEBACK_CHARS = 1200

ABLATIONS = [
    AblationMode.NONE,
]

SANDBOX_URL = os.getenv("SANDBOX_URL", "http://localhost:8000")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "github" / "ablation" / "patched_code"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PATCH_PROMPT_DIR = PROJECT_ROOT / "src/reprodbench/llm/prompts/code_generation/patched"
JUDGE_PROMPT_DIR = PROJECT_ROOT / "src/reprodbench/llm/prompts/judge_llm/patched"

MODEL_DATA_PATHS = {
    # "phi4:latest": PROJECT_ROOT
    # / "data"
    # / "github"
    # / "phi4_latest"
    # / "patched_code_gen.csv",
    
}

MODEL_SLUGS = {
    "phi4:latest": "phi4_latest",
    "gemma3:27b": "gemma3_27b",
    "qwen2.5-coder:32b": "qwen2.5-coder_32b",
    "deepseek-coder-v2:16b": "deepseek-coder-v2_16b",
}


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------


def now_ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def unpack_token_usage(usage: dict | None):
    if not usage:
        return None, None, None
    return (
        usage.get("prompt_tokens") or usage.get("input_tokens"),
        usage.get("completion_tokens") or usage.get("output_tokens"),
        usage.get("total_tokens"),
    )


def get_optional(row, col):
    if col not in row.index:
        return None
    v = row[col]
    return None if pd.isna(v) else str(v)


def safe_execute(
    executor, code: str, requirements: str, python_version: str
) -> ExecutionResult:
    try:
        return executor.execute_python(
            code=code,
            requirements=requirements,
            python_version=python_version,
            timeout_seconds=TIMEOUT_SECONDS,
        )
    except Exception as e:
        return ExecutionResult(
            stdout="",
            stderr="",
            exit_code=-1,
            runtime=None,
            timeout=False,
            image=None,
            docker_available=False,
            error=str(e),
        )


# ----------------------------------------------------------------------
# RUN PER MODEL
# ----------------------------------------------------------------------


def run_for_model(MODEL_NAME: str, DATA_PATH: Path):
    MODEL_SLUG = MODEL_SLUGS[MODEL_NAME]
    RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_PATH = (
        RESULTS_DIR
        / f"patched_code_generation_ablation_{MODEL_SLUG}_{JUDGE_MODEL_NAME}_{RUN_TS}.csv"
    )

    log_section(
        logger,
        "CONFIG",
        {
            "MODEL_NAME": MODEL_NAME,
            "MODEL_SLUG": MODEL_SLUG,
            "JUDGE_MODEL_NAME": JUDGE_MODEL_NAME,
            "SANDBOX_URL": SANDBOX_URL,
            "MAX_GENERATOR_RETRIES": MAX_GENERATOR_RETRIES,
            "MAX_REFINER_RETRIES": MAX_REFINER_RETRIES,
            "MAX_EXEC_RETRIES": MAX_EXEC_RETRIES,
            "MAX_JUDGE_RETRIES": MAX_JUDGE_RETRIES,
            "MAX_CYCLES": MAX_CYCLES,
            "TIMEOUT_SECONDS": TIMEOUT_SECONDS,
            "DATA_PATH": str(DATA_PATH),
            "RESULTS_PATH": str(RESULTS_PATH),
            "ABLATIONS": [a.name for a in ABLATIONS],
        },
    )

    dataset = pd.read_csv(DATA_PATH)
    executor = SandboxExecutorClient(SANDBOX_URL)

    generator = PatchedCodeGenerator(
        prompt_dir=PATCH_PROMPT_DIR,
        provider="ollama",
        model=MODEL_NAME,
        temperature=1.0,
    )

    refiner = PatchedCodeRefiner(
        prompt_dir=PATCH_PROMPT_DIR,
        provider="ollama",
        model=MODEL_NAME,
        temperature=1.0,
    )

    judge = PatchedCodeJudge(
        prompt_dir=JUDGE_PROMPT_DIR,
        provider="ollama",
        model=JUDGE_MODEL_NAME,
        temperature=1.0,
    )

    write_header = not RESULTS_PATH.exists()

    def write_row(row: dict):
        nonlocal write_header
        pd.DataFrame([row]).to_csv(
            RESULTS_PATH,
            mode="a",
            index=False,
            header=write_header,
        )
        write_header = False

    def safe_refine_exec_error(
        *,
        artifact,
        session_id: str,
        question: str,
        answer: str,
        stdout: str,
        stderr: str,
        docker_error: str,
        row_meta: dict,
    ):
        last_err = None
        for refine_attempt in range(1, MAX_REFINER_RETRIES + 1):
            try:
                new_artifact = refiner.refine_exec_error(
                    artifact=artifact,
                    question=question,
                    answer=answer,
                    stdout=stdout,
                    stderr=stderr,
                    docker_error=docker_error,
                    session_id=session_id,
                )
                return new_artifact, None
            except StructuredOutputError as e:
                last_err = str(e)
                write_row(
                    {
                        **row_meta,
                        "attempt": refine_attempt,
                        "stage": "refiner_exec_structured_output_error",
                        "python_version": getattr(artifact, "python_version", None),
                        "requirements": getattr(artifact, "requirements", None),
                        "buggy_code": getattr(artifact, "buggy_code", None),
                        "patched_code": getattr(artifact, "patched_code", None),
                        "generator_latency": None,
                        "generator_prompt_tokens": None,
                        "generator_completion_tokens": None,
                        "generator_total_tokens": None,
                        "judge_latency": None,
                        "judge_prompt_tokens": None,
                        "judge_completion_tokens": None,
                        "judge_total_tokens": None,
                        "stdout": stdout,
                        "stderr": stderr,
                        "exit_code": None,
                        "runtime": None,
                        "timeout": False,
                        "image": None,
                        "docker_available": None,
                        "error": last_err,
                        "judge_label": None,
                        "judge_rationale": None,
                        "attempt_ts": now_ts(),
                    }
                )
            except Exception as e:
                last_err = str(e)
                write_row(
                    {
                        **row_meta,
                        "attempt": refine_attempt,
                        "stage": "refiner_exec_error",
                        "python_version": getattr(artifact, "python_version", None),
                        "requirements": getattr(artifact, "requirements", None),
                        "buggy_code": getattr(artifact, "buggy_code", None),
                        "patched_code": getattr(artifact, "patched_code", None),
                        "generator_latency": None,
                        "generator_prompt_tokens": None,
                        "generator_completion_tokens": None,
                        "generator_total_tokens": None,
                        "judge_latency": None,
                        "judge_prompt_tokens": None,
                        "judge_completion_tokens": None,
                        "judge_total_tokens": None,
                        "stdout": stdout,
                        "stderr": stderr,
                        "exit_code": None,
                        "runtime": None,
                        "timeout": False,
                        "image": None,
                        "docker_available": None,
                        "error": last_err,
                        "judge_label": None,
                        "judge_rationale": None,
                        "attempt_ts": now_ts(),
                    }
                )
        return None, last_err

    def safe_refine_judge_mismatch(
        *,
        artifact,
        session_id: str,
        question: str,
        answer: str,
        stdout: str,
        stderr: str,
        judge_label: str,
        judge_rationale: str,
        row_meta: dict,
    ):
        last_err = None
        for refine_attempt in range(1, MAX_REFINER_RETRIES + 1):
            try:
                new_artifact = refiner.refine_judge_mismatch(
                    artifact=artifact,
                    question=question,
                    answer=answer,
                    stdout=stdout,
                    stderr=stderr,
                    judge_label=judge_label,
                    judge_rationale=judge_rationale,
                    session_id=session_id,
                )
                return new_artifact, None
            except StructuredOutputError as e:
                last_err = str(e)
                write_row(
                    {
                        **row_meta,
                        "attempt": refine_attempt,
                        "stage": "refiner_judge_structured_output_error",
                        "python_version": getattr(artifact, "python_version", None),
                        "requirements": getattr(artifact, "requirements", None),
                        "buggy_code": getattr(artifact, "buggy_code", None),
                        "patched_code": getattr(artifact, "patched_code", None),
                        "generator_latency": None,
                        "generator_prompt_tokens": None,
                        "generator_completion_tokens": None,
                        "generator_total_tokens": None,
                        "judge_latency": None,
                        "judge_prompt_tokens": None,
                        "judge_completion_tokens": None,
                        "judge_total_tokens": None,
                        "stdout": stdout,
                        "stderr": stderr,
                        "exit_code": None,
                        "runtime": None,
                        "timeout": False,
                        "image": None,
                        "docker_available": None,
                        "error": last_err,
                        "judge_label": judge_label,
                        "judge_rationale": judge_rationale,
                        "attempt_ts": now_ts(),
                    }
                )
            except Exception as e:
                last_err = str(e)
                write_row(
                    {
                        **row_meta,
                        "attempt": refine_attempt,
                        "stage": "refiner_judge_error",
                        "python_version": getattr(artifact, "python_version", None),
                        "requirements": getattr(artifact, "requirements", None),
                        "buggy_code": getattr(artifact, "buggy_code", None),
                        "patched_code": getattr(artifact, "patched_code", None),
                        "generator_latency": None,
                        "generator_prompt_tokens": None,
                        "generator_completion_tokens": None,
                        "generator_total_tokens": None,
                        "judge_latency": None,
                        "judge_prompt_tokens": None,
                        "judge_completion_tokens": None,
                        "judge_total_tokens": None,
                        "stdout": stdout,
                        "stderr": stderr,
                        "exit_code": None,
                        "runtime": None,
                        "timeout": False,
                        "image": None,
                        "docker_available": None,
                        "error": last_err,
                        "judge_label": judge_label,
                        "judge_rationale": judge_rationale,
                        "attempt_ts": now_ts(),
                    }
                )
        return None, last_err

    dataset["question_title"] = ""
    dataset["question_link"] = ""
    # dataset["question_body"] = dataset["question"]
    for ablation in ABLATIONS:
        # dataset already loaded from patched_code_gen.csv
        for idx, row in dataset.iterrows():
            print("\n" + "=" * 100)
            print(
                f"[{idx + 1}/{len(dataset)}] Model: {MODEL_NAME} Question ID: {row.question_id} Ablation: {ablation.name}"
            )
            print("=" * 100)

            question = row.question_body
            answer = row.accepted_answer_body

            ci = get_optional(row, "patched_code_intent")
            fr = get_optional(row, "patched_functional_requirements")
            scot = get_optional(row, "patched_scot")

            session_id = f"q_{row.question_id}_{MODEL_SLUG}_{ablation.name}"

            semantic_context = build_patched_semantic_context(
                ablation=ablation,
                ci=ci,
                fr=fr,
                scot=scot,
            )

            artifact = None

            row_meta = {
                "question_id": row.question_id,
                "question_link": getattr(row, "question_link", None),
                "model_name": MODEL_NAME,
                "model_slug": MODEL_SLUG,
                "ablation": ablation.name,
            }

            # ----------------------------------------------------------
            # GENERATOR
            # ----------------------------------------------------------
            for gen_attempt in range(1, MAX_GENERATOR_RETRIES + 1):
                try:
                    artifact = generator.generate(
                        question=question,
                        answer=answer,
                        buggy_code=row.buggy_code,
                        python_version=row.python_version,
                        requirements=row.requirements,
                        semantic_context=semantic_context,
                        session_id=session_id,
                    )

                    g_prompt, g_comp, g_total = unpack_token_usage(
                        getattr(artifact, "token_usage", None)
                    )

                    write_row(
                        {
                            **row_meta,
                            "cycle": 0,
                            "attempt": gen_attempt,
                            "stage": "generator",
                            "python_version": artifact.python_version,
                            "requirements": artifact.requirements,
                            "buggy_code": getattr(
                                artifact, "buggy_code", row.buggy_code
                            ),
                            "patched_code": artifact.patched_code,
                            "generator_latency": getattr(artifact, "latency", None),
                            "generator_prompt_tokens": g_prompt,
                            "generator_completion_tokens": g_comp,
                            "generator_total_tokens": g_total,
                            "judge_latency": None,
                            "judge_prompt_tokens": None,
                            "judge_completion_tokens": None,
                            "judge_total_tokens": None,
                            "stdout": "",
                            "stderr": "",
                            "exit_code": None,
                            "runtime": None,
                            "timeout": False,
                            "image": None,
                            "docker_available": None,
                            "error": None,
                            "judge_label": None,
                            "judge_rationale": None,
                            "attempt_ts": now_ts(),
                        }
                    )
                    break

                except StructuredOutputError as e:
                    write_row(
                        {
                            **row_meta,
                            "cycle": 0,
                            "attempt": gen_attempt,
                            "stage": "generator_error",
                            "python_version": None,
                            "requirements": None,
                            "buggy_code": row.buggy_code,
                            "patched_code": None,
                            "generator_latency": None,
                            "generator_prompt_tokens": None,
                            "generator_completion_tokens": None,
                            "generator_total_tokens": None,
                            "judge_latency": None,
                            "judge_prompt_tokens": None,
                            "judge_completion_tokens": None,
                            "judge_total_tokens": None,
                            "stdout": "",
                            "stderr": "",
                            "exit_code": None,
                            "runtime": None,
                            "timeout": False,
                            "image": None,
                            "docker_available": None,
                            "error": str(e),
                            "judge_label": None,
                            "judge_rationale": None,
                            "attempt_ts": now_ts(),
                        }
                    )
                except Exception as e:
                    write_row(
                        {
                            **row_meta,
                            "cycle": 0,
                            "attempt": gen_attempt,
                            "stage": "generator_exception",
                            "python_version": None,
                            "requirements": None,
                            "buggy_code": row.buggy_code,
                            "patched_code": None,
                            "generator_latency": None,
                            "generator_prompt_tokens": None,
                            "generator_completion_tokens": None,
                            "generator_total_tokens": None,
                            "judge_latency": None,
                            "judge_prompt_tokens": None,
                            "judge_completion_tokens": None,
                            "judge_total_tokens": None,
                            "stdout": "",
                            "stderr": "",
                            "exit_code": None,
                            "runtime": None,
                            "timeout": False,
                            "image": None,
                            "docker_available": None,
                            "error": str(e),
                            "judge_label": None,
                            "judge_rationale": None,
                            "attempt_ts": now_ts(),
                        }
                    )

            if artifact is None:
                continue

            # ----------------------------------------------------------
            # CYCLES
            # ----------------------------------------------------------
            for cycle in range(1, MAX_CYCLES + 1):
                exec_success = False
                exec_result = None

                for attempt in range(1, MAX_EXEC_RETRIES + 1):
                    log_step(
                        logger,
                        f"[{MODEL_NAME}] [{ablation.name}] cycle: {cycle} attempt: {attempt}",
                    )

                    exec_result = safe_execute(
                        executor=executor,
                        code=artifact.patched_code,
                        requirements=artifact.requirements,
                        python_version=artifact.python_version,
                    )

                    log_section(
                        logger,
                        "Generator Output",
                        {
                            "patched_code": preview(artifact.patched_code, 600),
                            "requirements": artifact.requirements,
                            "python_version": artifact.python_version,
                        },
                    )
                    log_section(logger, "Exec Output", {"exec": exec_result})

                    write_row(
                        {
                            **row_meta,
                            "cycle": cycle,
                            "attempt": attempt,
                            "stage": "exec",
                            "python_version": artifact.python_version,
                            "requirements": artifact.requirements,
                            "buggy_code": getattr(
                                artifact, "buggy_code", row.buggy_code
                            ),
                            "patched_code": artifact.patched_code,
                            "generator_latency": None,
                            "generator_prompt_tokens": None,
                            "generator_completion_tokens": None,
                            "generator_total_tokens": None,
                            "judge_latency": None,
                            "judge_prompt_tokens": None,
                            "judge_completion_tokens": None,
                            "judge_total_tokens": None,
                            "stdout": exec_result.stdout,
                            "stderr": exec_result.stderr,
                            "exit_code": exec_result.exit_code,
                            "runtime": exec_result.runtime,
                            "timeout": exec_result.timeout,
                            "image": exec_result.image,
                            "docker_available": exec_result.docker_available,
                            "error": exec_result.error,
                            "judge_label": None,
                            "judge_rationale": None,
                            "attempt_ts": now_ts(),
                        }
                    )

                    if exec_result.exit_code == 0:
                        exec_success = True
                        break

                    refined, ref_err = safe_refine_exec_error(
                        artifact=artifact,
                        session_id=session_id,
                        question=question,
                        answer=answer,
                        stdout=exec_result.stdout,
                        stderr=preview(exec_result.stderr, MAX_TRACEBACK_CHARS),
                        docker_error=preview(exec_result.error, MAX_DOCKER_ERROR_CHARS),
                        row_meta={**row_meta, "cycle": cycle},
                    )

                    log_section(logger, "Refiner Output", {"refined": refined})

                    if refined is None:
                        write_row(
                            {
                                **row_meta,
                                "cycle": cycle,
                                "attempt": attempt,
                                "stage": "refiner_exec_failed",
                                "python_version": artifact.python_version,
                                "requirements": artifact.requirements,
                                "buggy_code": getattr(
                                    artifact, "buggy_code", row.buggy_code
                                ),
                                "patched_code": artifact.patched_code,
                                "generator_latency": None,
                                "generator_prompt_tokens": None,
                                "generator_completion_tokens": None,
                                "generator_total_tokens": None,
                                "judge_latency": None,
                                "judge_prompt_tokens": None,
                                "judge_completion_tokens": None,
                                "judge_total_tokens": None,
                                "stdout": exec_result.stdout,
                                "stderr": exec_result.stderr,
                                "exit_code": exec_result.exit_code,
                                "runtime": exec_result.runtime,
                                "timeout": exec_result.timeout,
                                "image": exec_result.image,
                                "docker_available": exec_result.docker_available,
                                "error": ref_err,
                                "judge_label": None,
                                "judge_rationale": None,
                                "attempt_ts": now_ts(),
                            }
                        )
                        break

                    artifact = refined

                if not exec_success:
                    break

                judge_result = None
                judge_err = None

                for judge_attempt in range(1, MAX_JUDGE_RETRIES + 1):
                    try:
                        judge_result = judge.judge(
                            question=question,
                            answer=answer,
                            patched_code=artifact.patched_code,
                            stdout=exec_result.stdout,
                            stderr=exec_result.stderr,
                            exit_code=exec_result.exit_code,
                        )
                        log_section(
                            logger,
                            "Judge",
                            {
                                "cycle": cycle,
                                "attempt": judge_attempt,
                                "judge_label": judge_result.label,
                                "judge_rationale": judge_result.rationale,
                            },
                        )
                        judge_err = None
                        break
                    except Exception as e:
                        judge_err = str(e)
                        write_row(
                            {
                                **row_meta,
                                "cycle": cycle,
                                "attempt": judge_attempt,
                                "stage": "judge_error",
                                "python_version": artifact.python_version,
                                "requirements": artifact.requirements,
                                "buggy_code": getattr(
                                    artifact, "buggy_code", row.buggy_code
                                ),
                                "patched_code": artifact.patched_code,
                                "generator_latency": None,
                                "generator_prompt_tokens": None,
                                "generator_completion_tokens": None,
                                "generator_total_tokens": None,
                                "judge_latency": None,
                                "judge_prompt_tokens": None,
                                "judge_completion_tokens": None,
                                "judge_total_tokens": None,
                                "stdout": exec_result.stdout,
                                "stderr": exec_result.stderr,
                                "exit_code": exec_result.exit_code,
                                "runtime": exec_result.runtime,
                                "timeout": exec_result.timeout,
                                "image": exec_result.image,
                                "docker_available": exec_result.docker_available,
                                "error": judge_err,
                                "judge_label": None,
                                "judge_rationale": None,
                                "attempt_ts": now_ts(),
                            }
                        )

                if judge_result is None:
                    break

                j_prompt, j_comp, j_total = unpack_token_usage(
                    getattr(judge_result, "token_usage", None)
                )

                write_row(
                    {
                        **row_meta,
                        "cycle": cycle,
                        "attempt": None,
                        "stage": "judge",
                        "python_version": artifact.python_version,
                        "requirements": artifact.requirements,
                        "buggy_code": getattr(artifact, "buggy_code", row.buggy_code),
                        "patched_code": artifact.patched_code,
                        "generator_latency": None,
                        "generator_prompt_tokens": None,
                        "generator_completion_tokens": None,
                        "generator_total_tokens": None,
                        "judge_latency": getattr(judge_result, "latency", None),
                        "judge_prompt_tokens": j_prompt,
                        "judge_completion_tokens": j_comp,
                        "judge_total_tokens": j_total,
                        "stdout": exec_result.stdout,
                        "stderr": exec_result.stderr,
                        "exit_code": exec_result.exit_code,
                        "runtime": exec_result.runtime,
                        "timeout": exec_result.timeout,
                        "image": exec_result.image,
                        "docker_available": exec_result.docker_available,
                        "error": None,
                        "judge_label": judge_result.label,
                        "judge_rationale": judge_result.rationale,
                        "attempt_ts": now_ts(),
                    }
                )

                if judge_result.label == "correct":
                    break

                refined, ref_err = safe_refine_judge_mismatch(
                    artifact=artifact,
                    session_id=session_id,
                    question=question,
                    answer=answer,
                    stdout=exec_result.stdout,
                    stderr=exec_result.stderr,
                    judge_label=judge_result.label,
                    judge_rationale=judge_result.rationale,
                    row_meta={**row_meta, "cycle": cycle},
                )

                log_section(
                    logger,
                    "Judge Mismatch Refined",
                    {"cycle": cycle, "refined": refined, "refined_err": ref_err},
                )

                if refined is None:
                    write_row(
                        {
                            **row_meta,
                            "cycle": cycle,
                            "attempt": None,
                            "stage": "refiner_judge_failed",
                            "python_version": artifact.python_version,
                            "requirements": artifact.requirements,
                            "buggy_code": getattr(
                                artifact, "buggy_code", row.buggy_code
                            ),
                            "patched_code": artifact.patched_code,
                            "generator_latency": None,
                            "generator_prompt_tokens": None,
                            "generator_completion_tokens": None,
                            "generator_total_tokens": None,
                            "judge_latency": None,
                            "judge_prompt_tokens": None,
                            "judge_completion_tokens": None,
                            "judge_total_tokens": None,
                            "stdout": exec_result.stdout,
                            "stderr": exec_result.stderr,
                            "exit_code": exec_result.exit_code,
                            "runtime": exec_result.runtime,
                            "timeout": exec_result.timeout,
                            "image": exec_result.image,
                            "docker_available": exec_result.docker_available,
                            "error": ref_err,
                            "judge_label": judge_result.label,
                            "judge_rationale": judge_result.rationale,
                            "attempt_ts": now_ts(),
                        }
                    )
                    break

                artifact = refined

    log_step(logger, f"[{MODEL_NAME}] PATCHED RESULTS WRITTEN → {RESULTS_PATH}")


# ----------------------------------------------------------------------
# RUN ALL MODELS
# ----------------------------------------------------------------------

for MODEL_NAME, DATA_PATH in MODEL_DATA_PATHS.items():
    run_for_model(MODEL_NAME, DATA_PATH)
