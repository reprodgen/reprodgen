import re
from datetime import datetime
from pprint import pprint

import pandas as pd
from langchain_core.messages import BaseMessage


def strip_code_fences(code: str) -> str:
    code = code.strip()

    # Remove opening fence: ``` or ```python
    code = re.sub(r"^```[\w+-]*\n", "", code)

    # Remove closing fence
    code = re.sub(r"\n```$", "", code)

    return code.strip()


DEBUG_PROMPTS = False  # flip to False later


def debug_prompt(x):
    """
    Debug runnable that safely prints whatever LangChain passes
    (messages, dict, or string) WITHOUT breaking the chain.
    """
    if not DEBUG_PROMPTS:
        return x

    print("\n" + "=" * 100)
    print("FULLY RENDERED INPUT TO NEXT STAGE")
    print("=" * 100)

    # Case 1: list of messages (ideal case)
    if isinstance(x, list):
        for msg in x:
            if isinstance(msg, BaseMessage):
                role = msg.type.upper()
                print(f"\n[{role}]\n{msg.content}")
            else:
                print("\n[UNKNOWN MESSAGE TYPE]")
                pprint(msg)

    # Case 2: single BaseMessage
    elif isinstance(x, BaseMessage):
        print(f"\n[{x.type.upper()}]\n{x.content}")

    # Case 3: dict (already rendered variables)
    elif isinstance(x, dict):
        print("\n[DICT INPUT]")
        pprint(x)

    # Case 4: string or anything else
    else:
        print("\n[RAW INPUT]")
        print(x)

    print("=" * 100 + "\n")
    return x


def now_timestamp(timespec="seconds"):
    return datetime.now().isoformat(timespec=timespec)


def unpack_token_usage(usage: dict | None):
    if not usage:
        return None, None, None
    return (
        usage.get("prompt_tokens") or usage.get("input_tokens"),
        usage.get("completion_tokens") or usage.get("output_tokens"),
        usage.get("total_tokens"),
    )


def get_optional(row, col: str):
    if col not in row.index:
        return None
    v = row[col]
    return None if pd.isna(v) else str(v)
