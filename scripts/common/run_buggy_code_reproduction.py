import os
import pprint
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from reprodbench.ablation.builder import build_buggy_semantic_context
from reprodbench.ablation.mode import AblationMode
from reprodbench.executor.sandbox_client import ExecutionResult, SandboxExecutorClient
from reprodbench.pipeline.buggy_code import (
    BuggyCodeGenerator,
    BuggyCodeRefiner,
    StructuredOutputError,
)
from reprodbench.pipeline.judge_buggy_code import BuggyCodeJudge
from reprodbench.utils.logger import log_section, log_step, setup_logger
from reprodbench.utils.text import preview

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

load_dotenv()
logger = setup_logger("reprodbench.pipeline")

MAX_GENERATOR_RETRIES = 3
MAX_REFINER_RETRIES = 3
MAX_EXEC_RETRIES = 3
MAX_JUDGE_RETRIES = 3
MAX_CYCLES = 3
TIMEOUT_SECONDS = 120

GEN_MODELS = [
    # "phi4:latest",
    # "gemma3:27b",
    # "qwen2.5-coder:32b",
    # "deepseek-coder-v2:16b",
    # "codellama:7b",
    "gpt-oss:20b"
]
JUDGE_MODEL_NAME = "qwen3:8b"

MAX_DOCKER_ERROR_CHARS = 800
MAX_TRACEBACK_CHARS = 1200

ABLATIONS = [
    AblationMode.CI_FR_SCOT,
    # AblationMode.CI_FR,
    # AblationMode.FR_SCOT,
    # AblationMode.CI_SCOT,
    # AblationMode.NONE,
]

SANDBOX_URL = os.getenv("SANDBOX_URL", "http://localhost:8000")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "github" / "buggy_code"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")

MODEL_DATA_PATHS = {
    # "phi4:latest": PROJECT_ROOT
    # / "data"
    # / "github"
    # / "phi4_latest"
    # / "buggy_code_gen.csv",
   
}

MODEL_SLUGS = {
    "phi4:latest": "phi4_latest",
    "gemma3:27b": "gemma3_27b",
    "qwen2.5-coder:32b": "qwen2.5-coder_32b",
    "deepseek-coder-v2:16b": "deepseek-coder-v2_16b",
    "codellama:7b": "codellama_7b",
    "gpt-oss:20b": "gpt-oss_20b",
}

GEN_PROMPT_DIR = PROJECT_ROOT / "src/reprodbench/llm/prompts/code_generation/buggy"
JUDGE_PROMPT_DIR = PROJECT_ROOT / "src/reprodbench/llm/prompts/judge_llm/buggy"

log_section(
    logger,
    "CONFIG",
    {
        "GEN_MODELS": GEN_MODELS,
        "JUDGE_MODEL_NAME": JUDGE_MODEL_NAME,
        "MODEL_DATA_PATHS": {k: str(v) for k, v in MODEL_DATA_PATHS.items()},
        "SANDBOX_URL": SANDBOX_URL,
        "MAX_GENERATOR_RETRIES": MAX_GENERATOR_RETRIES,
        "MAX_REFINER_RETRIES": MAX_REFINER_RETRIES,
        "MAX_EXEC_RETRIES": MAX_EXEC_RETRIES,
        "MAX_JUDGE_RETRIES": MAX_JUDGE_RETRIES,
        "MAX_CYCLES": MAX_CYCLES,
        "TIMEOUT_SECONDS": TIMEOUT_SECONDS,
        "ABLATIONS": [a.name for a in ABLATIONS],
    },
)

# ----------------------------------------------------------------------
# INIT
# ----------------------------------------------------------------------

executor = SandboxExecutorClient(SANDBOX_URL)

RESULTS_PATH = None
write_header = False

# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------


def now_ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_row(row: dict):
    global write_header, RESULTS_PATH
    pd.DataFrame([row]).to_csv(
        RESULTS_PATH,
        mode="a",
        index=False,
        header=write_header,
    )
    write_header = False


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


def safe_execute(code: str, requirements: str, python_version: str) -> ExecutionResult:
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


import re


def normalize_python_version(v: str | None) -> str:
    if not v:
        return "3.10"
    m = re.search(r"(\d+)\.(\d+)", str(v))
    if not m:
        return "3.10"
    return f"{m.group(1)}.{m.group(2)}"


def safe_refine_exec_error(
    *, artifact, session_id: str, buggy_stderr: str, docker_error: str
):
    """
    Protects exec-error refiner with retries against StructuredOutputError
    and writes a row for each refiner failure.
    Returns: (new_artifact or None, last_error_str or None)
    """
    last_err = None
    for refine_attempt in range(1, MAX_REFINER_RETRIES + 1):
        try:
            new_artifact = refiner.refine(
                artifact=artifact,
                buggy_stderr=buggy_stderr,
                docker_error=docker_error,
                session_id=session_id,
            )
            return new_artifact, None
        except StructuredOutputError as e:
            last_err = str(e)
            write_row(
                {
                    "question_id": None,
                    "question_link": None,
                    "ablation": None,
                    "cycle": None,
                    "attempt": refine_attempt,
                    "stage": "refiner_exec_structured_output_error",
                    "python_version": getattr(artifact, "python_version", None),
                    "requirements": getattr(artifact, "requirements", None),
                    "buggy_code": getattr(artifact, "buggy_code", None),
                    "generator_model": MODEL_NAME,
                    "judge_model": JUDGE_MODEL_NAME,
                    "generator_latency": None,
                    "generator_prompt_tokens": None,
                    "generator_completion_tokens": None,
                    "generator_total_tokens": None,
                    "judge_latency": None,
                    "judge_prompt_tokens": None,
                    "judge_completion_tokens": None,
                    "judge_total_tokens": None,
                    "stdout": "",
                    "stderr": buggy_stderr,
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
                    "question_id": None,
                    "question_link": None,
                    "ablation": None,
                    "cycle": None,
                    "attempt": refine_attempt,
                    "stage": "refiner_exec_error",
                    "python_version": getattr(artifact, "python_version", None),
                    "requirements": getattr(artifact, "requirements", None),
                    "buggy_code": getattr(artifact, "buggy_code", None),
                    "generator_model": MODEL_NAME,
                    "judge_model": JUDGE_MODEL_NAME,
                    "generator_latency": None,
                    "generator_prompt_tokens": None,
                    "generator_completion_tokens": None,
                    "generator_total_tokens": None,
                    "judge_latency": None,
                    "judge_prompt_tokens": None,
                    "judge_completion_tokens": None,
                    "judge_total_tokens": None,
                    "stdout": "",
                    "stderr": buggy_stderr,
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
    judge_label: str,
    judge_rationale: str,
    stdout: str,
    stderr: str,
):
    """
    Calls a judge-mismatch refiner if it exists; protects it with retries.
    Returns: (new_artifact or None, last_error_str or None)
    """
    if not hasattr(refiner, "refine_judge_mismatch"):
        return None, "BuggyCodeRefiner has no method refine_judge_mismatch(...)"

    last_err = None
    for refine_attempt in range(1, MAX_REFINER_RETRIES + 1):
        try:
            new_artifact = refiner.refine_judge_mismatch(
                artifact=artifact,
                question=question,
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
                    "question_id": None,
                    "question_link": None,
                    "ablation": None,
                    "cycle": None,
                    "attempt": refine_attempt,
                    "stage": "refiner_judge_structured_output_error",
                    "python_version": getattr(artifact, "python_version", None),
                    "requirements": getattr(artifact, "requirements", None),
                    "buggy_code": getattr(artifact, "buggy_code", None),
                    "generator_model": MODEL_NAME,
                    "judge_model": JUDGE_MODEL_NAME,
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
                    "question_id": None,
                    "question_link": None,
                    "ablation": None,
                    "cycle": None,
                    "attempt": refine_attempt,
                    "stage": "refiner_judge_error",
                    "python_version": getattr(artifact, "python_version", None),
                    "requirements": getattr(artifact, "requirements", None),
                    "buggy_code": getattr(artifact, "buggy_code", None),
                    "generator_model": MODEL_NAME,
                    "judge_model": JUDGE_MODEL_NAME,
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


# ----------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------

for MODEL_NAME in GEN_MODELS:
    DATA_PATH = MODEL_DATA_PATHS[MODEL_NAME]
    MODEL_SLUG = MODEL_SLUGS[MODEL_NAME]

    RESULTS_PATH = (
        RESULTS_DIR
        / f"buggy_code_generation_ablation_{MODEL_SLUG}_{JUDGE_MODEL_NAME}_{RUN_TS}.csv"
    )
    write_header = not RESULTS_PATH.exists()

    dataset = pd.read_csv(DATA_PATH)

    generator = BuggyCodeGenerator(
        prompt_dir=GEN_PROMPT_DIR,
        provider="ollama",
        model=MODEL_NAME,
        temperature=1,
    )

    refiner = BuggyCodeRefiner(
        prompt_dir=GEN_PROMPT_DIR,
        provider="ollama",
        model=MODEL_NAME,
        temperature=1,
    )

    judge = BuggyCodeJudge(
        prompt_dir=JUDGE_PROMPT_DIR,
        provider="ollama",
        model=JUDGE_MODEL_NAME,
        temperature=1,
    )

    log_section(
        logger,
        "MODEL RUN",
        {
            "MODEL_NAME": MODEL_NAME,
            "DATA_PATH": str(DATA_PATH),
            "RESULTS_PATH": str(RESULTS_PATH),
            "NUM_ROWS": len(dataset),
        },
    )
    dataset["question_title"] = ""
    dataset["question_link"] = ""
    # dataset["question_body"] = dataset["question"]
    # dataset["question_body"] = dataset["question"]

    # dataset = dataset.iloc[52:]  # Limit for testing
    for ablation in ABLATIONS:
        for idx, row in dataset.iterrows():
            print("\n" + "=" * 100)
            print(
                f"[{MODEL_NAME}] [{idx + 1}/{len(dataset)}] Question ID: {row.question_id} Ablation: {ablation}"
            )
            print("=" * 100)

            question = row.question_title + "\n" + row.question_body

            ci = get_optional(row, "buggy_code_intent")
            fr = get_optional(row, "buggy_functional_requirements")
            scot = get_optional(row, "buggy_scot")

            session_id = f"{MODEL_SLUG}_q_{row.question_id}_{ablation.name}"

            semantic_context = build_buggy_semantic_context(
                ablation=ablation,
                ci=ci,
                fr=fr,
                scot=scot,
            )

            artifact = None

            # --------------------------------------------------------------
            # GENERATOR
            # --------------------------------------------------------------
            for gen_attempt in range(1, MAX_GENERATOR_RETRIES + 1):
                try:
                    artifact = generator.generate(
                        question_text=question,
                        semantic_context=semantic_context,
                        session_id=session_id,
                    )

                    g_prompt, g_comp, g_total = unpack_token_usage(artifact.token_usage)

                    write_row(
                        {
                            "question_id": row.question_id,
                            "question_link": row.question_link,
                            "ablation": ablation.name,
                            "cycle": 0,
                            "attempt": gen_attempt,
                            "stage": "generator",
                            "generator_model": MODEL_NAME,
                            "judge_model": JUDGE_MODEL_NAME,
                            "python_version": artifact.python_version,
                            "requirements": artifact.requirements,
                            "buggy_code": artifact.buggy_code,
                            "generator_latency": artifact.latency,
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
                            "question_id": row.question_id,
                            "question_link": row.question_link,
                            "ablation": ablation.name,
                            "cycle": 0,
                            "attempt": gen_attempt,
                            "stage": "generator_error",
                            "generator_model": MODEL_NAME,
                            "judge_model": JUDGE_MODEL_NAME,
                            "python_version": None,
                            "requirements": None,
                            "buggy_code": None,
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
                            "question_id": row.question_id,
                            "question_link": row.question_link,
                            "ablation": ablation.name,
                            "cycle": 0,
                            "attempt": gen_attempt,
                            "stage": "generator_exception",
                            "generator_model": MODEL_NAME,
                            "judge_model": JUDGE_MODEL_NAME,
                            "python_version": None,
                            "requirements": None,
                            "buggy_code": None,
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

            # --------------------------------------------------------------
            # CYCLES: EXEC → (refine exec errors) → JUDGE → (refine mismatch)
            # --------------------------------------------------------------
            for cycle in range(1, MAX_CYCLES + 1):
                exec_success = False
                exec_result = None

                for attempt in range(1, MAX_EXEC_RETRIES + 1):
                    log_step(
                        logger, f"model: {MODEL_NAME} cycle: {cycle} attempt: {attempt}"
                    )

                    exec_result = safe_execute(
                        code=artifact.buggy_code,
                        requirements=artifact.requirements,
                        python_version=artifact.python_version,
                    )

                    log_section(
                        logger,
                        "Generator Output",
                        {
                            "code": artifact.buggy_code,
                            "requirements": artifact.requirements,
                            "python_version": artifact.python_version,
                        },
                    )
                    log_section(logger, "Exec Output", {"exec": exec_result})

                    write_row(
                        {
                            "question_id": row.question_id,
                            "question_link": row.question_link,
                            "ablation": ablation.name,
                            "cycle": cycle,
                            "attempt": attempt,
                            "stage": "exec",
                            "generator_model": MODEL_NAME,
                            "judge_model": JUDGE_MODEL_NAME,
                            "python_version": artifact.python_version,
                            "requirements": artifact.requirements,
                            "buggy_code": artifact.buggy_code,
                            "stdout": exec_result.stdout,
                            "stderr": exec_result.stderr,
                            "exit_code": exec_result.exit_code,
                            "runtime": exec_result.runtime,
                            "timeout": exec_result.timeout,
                            "image": exec_result.image,
                            "docker_available": exec_result.docker_available,
                            "error": exec_result.error,
                            "attempt_ts": now_ts(),
                        }
                    )

                    if exec_result.exit_code == 0:
                        exec_success = True
                        break

                    refined, ref_err = safe_refine_exec_error(
                        artifact=artifact,
                        session_id=session_id,
                        buggy_stderr=preview(exec_result.stderr, MAX_TRACEBACK_CHARS),
                        docker_error=preview(exec_result.error, MAX_DOCKER_ERROR_CHARS),
                    )

                    log_section(logger, "Refiner Output", {"refined": refined})

                    if refined is None:
                        write_row(
                            {
                                "question_id": row.question_id,
                                "question_link": row.question_link,
                                "ablation": ablation.name,
                                "cycle": cycle,
                                "attempt": attempt,
                                "stage": "refiner_exec_failed",
                                "generator_model": MODEL_NAME,
                                "judge_model": JUDGE_MODEL_NAME,
                                "python_version": artifact.python_version,
                                "requirements": artifact.requirements,
                                "buggy_code": artifact.buggy_code,
                                "stdout": exec_result.stdout,
                                "stderr": exec_result.stderr,
                                "exit_code": exec_result.exit_code,
                                "runtime": exec_result.runtime,
                                "timeout": exec_result.timeout,
                                "image": exec_result.image,
                                "docker_available": exec_result.docker_available,
                                "error": ref_err,
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
                            buggy_code=artifact.buggy_code,
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
                                "question_id": row.question_id,
                                "question_link": row.question_link,
                                "ablation": ablation.name,
                                "cycle": cycle,
                                "attempt": judge_attempt,
                                "stage": "judge_error",
                                "generator_model": MODEL_NAME,
                                "judge_model": JUDGE_MODEL_NAME,
                                "python_version": artifact.python_version,
                                "requirements": artifact.requirements,
                                "buggy_code": artifact.buggy_code,
                                "stdout": exec_result.stdout,
                                "stderr": exec_result.stderr,
                                "exit_code": exec_result.exit_code,
                                "runtime": exec_result.runtime,
                                "timeout": exec_result.timeout,
                                "image": exec_result.image,
                                "docker_available": exec_result.docker_available,
                                "error": judge_err,
                                "attempt_ts": now_ts(),
                            }
                        )

                if judge_result is None:
                    break

                j_prompt, j_comp, j_total = unpack_token_usage(judge_result.token_usage)

                write_row(
                    {
                        "question_id": row.question_id,
                        "question_link": row.question_link,
                        "ablation": ablation.name,
                        "cycle": cycle,
                        "attempt": None,
                        "stage": "judge",
                        "generator_model": MODEL_NAME,
                        "judge_model": JUDGE_MODEL_NAME,
                        "python_version": artifact.python_version,
                        "requirements": artifact.requirements,
                        "buggy_code": artifact.buggy_code,
                        "generator_latency": None,
                        "generator_prompt_tokens": None,
                        "generator_completion_tokens": None,
                        "generator_total_tokens": None,
                        "judge_latency": judge_result.latency,
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
                    judge_label=judge_result.label,
                    judge_rationale=judge_result.rationale,
                    stdout=exec_result.stdout,
                    stderr=exec_result.stderr,
                )
                log_section(
                    logger,
                    "Judge Mismatch Refined",
                    {"cycle": cycle, "refined": refined, "refined_err": ref_err},
                )

                if refined is None:
                    write_row(
                        {
                            "question_id": row.question_id,
                            "question_link": row.question_link,
                            "ablation": ablation.name,
                            "cycle": cycle,
                            "attempt": None,
                            "stage": "refiner_judge_failed",
                            "generator_model": MODEL_NAME,
                            "judge_model": JUDGE_MODEL_NAME,
                            "python_version": artifact.python_version,
                            "requirements": artifact.requirements,
                            "buggy_code": artifact.buggy_code,
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

    log_step(logger, f"[{MODEL_NAME}] RESULTS WRITTEN → {RESULTS_PATH}")
