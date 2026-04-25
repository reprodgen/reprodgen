import pandas as pd

QUESTION_ID_COL = "question_id"
JUDGE_LABEL_COL = "judge_label"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _select_last_attempt(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if QUESTION_ID_COL not in df.columns:
        raise ValueError(f"[{name}] Missing column: {QUESTION_ID_COL}")

    if "cycle" not in df.columns:
        raise ValueError(f"[{name}] Missing column: cycle")

    sort_cols = [QUESTION_ID_COL, "cycle"]
    ascending = [True, True]

    if "attempt" in df.columns:
        sort_cols.append("attempt")
        ascending.append(True)

    df = df.sort_values(by=sort_cols, ascending=ascending)
    out = df.groupby(QUESTION_ID_COL, as_index=False).tail(1)

    print(f"[{name}] last_attempt: {len(out)}/{len(df)}")
    return out


def _filter_judge_correct(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if JUDGE_LABEL_COL not in df.columns:
        raise ValueError(f"[{name}] Missing column: {JUDGE_LABEL_COL}")

    before = len(df)
    out = df[
        df[JUDGE_LABEL_COL].astype(str).str.strip().str.lower().eq("correct")
    ].copy()

    print(f"[{name}] judge_correct: {len(out)}/{before}")
    return out


def _assert_unique(df: pd.DataFrame, name: str):
    dup = df[df.duplicated(QUESTION_ID_COL)]
    if not dup.empty:
        raise ValueError(
            f"[{name}] Duplicate question_id detected:\n"
            f"{dup[QUESTION_ID_COL].value_counts()}"
        )


# ----------------------------------------------------------------------
# Buggy generation dataset
# ----------------------------------------------------------------------


def build_buggy_generation_dataset(
    *,
    df_base: pd.DataFrame,
    bci: pd.DataFrame,
    bfr: pd.DataFrame,
    bscot: pd.DataFrame,
) -> pd.DataFrame:
    bci = _filter_judge_correct(_select_last_attempt(bci, "BCI"), "BCI")
    bfr = _filter_judge_correct(_select_last_attempt(bfr, "BFR"), "BFR")
    bscot = _filter_judge_correct(_select_last_attempt(bscot, "BSCOT"), "BSCOT")

    bci = bci[[QUESTION_ID_COL, "buggy_code_intent"]]
    bfr = bfr[[QUESTION_ID_COL, "buggy_functional_requirements"]]
    bscot = bscot[[QUESTION_ID_COL, "buggy_scot"]]

    for name, df in {
        "BCI": bci,
        "BFR": bfr,
        "BSCOT": bscot,
        "BASE": df_base,
    }.items():
        if QUESTION_ID_COL not in df.columns:
            raise ValueError(f"[{name}] Missing column: {QUESTION_ID_COL}")
        _assert_unique(df, name)

    df_buggy = bci.merge(bfr, on=QUESTION_ID_COL, how="inner").merge(
        bscot, on=QUESTION_ID_COL, how="inner"
    )

    print(f"[MERGE] Buggy CI ∩ FR ∩ SCOT = {len(df_buggy)}")

    base_cols = [
        "question_id",
        "question_link",
        "question_title",
        "question_body",
        "accepted_answer_body",
    ]

    missing = set(base_cols) - set(df_base.columns)
    if missing:
        raise ValueError(f"[BASE] Missing required columns: {missing}")

    df_final = df_base[base_cols].merge(df_buggy, on=QUESTION_ID_COL, how="inner")
    print(f"[MERGE] Base ∩ Buggy = {len(df_final)}")

    return df_final


# ----------------------------------------------------------------------
# Patched guidance (buggy → patched inputs)
# ----------------------------------------------------------------------


# def build_patched_guidance_generation_dataset(
#     *,
#     buggy_base: pd.DataFrame,
#     buggy_gen: pd.DataFrame,
# ) -> pd.DataFrame:
#     buggy_correct = buggy_gen[
#         buggy_gen[JUDGE_LABEL_COL].astype(str).str.strip().str.lower().eq("correct")
#     ].copy()

#     buggy_correct = (
#         buggy_correct.sort_values([QUESTION_ID_COL, "cycle", "attempt"])
#         .groupby(QUESTION_ID_COL, as_index=False)
#         .tail(1)
#         .reset_index(drop=True)
#     )

#     if not buggy_correct[QUESTION_ID_COL].is_unique:
#         raise AssertionError("buggy_runs not unique per question_id")

#     buggy_correct = buggy_correct[
#         [
#             QUESTION_ID_COL,
#             "python_version",
#             "requirements",
#             "buggy_code",
#         ]
#     ]

#     final = buggy_base.merge(buggy_correct, on=QUESTION_ID_COL, how="inner")

#     if not final[QUESTION_ID_COL].is_unique:
#         raise AssertionError(
#             "Final patched guidance dataset not unique per question_id"
#         )

#     assert final.notnull().all().all()
#     return final


def build_patched_guidance_generation_dataset(
    *,
    buggy_base: pd.DataFrame,
    buggy_gen: pd.DataFrame,
) -> pd.DataFrame:
    buggy_correct = buggy_gen[
        buggy_gen[JUDGE_LABEL_COL].astype(str).str.strip().str.lower().eq("correct")
    ].copy()

    buggy_correct = (
        buggy_correct.sort_values([QUESTION_ID_COL, "cycle", "attempt"])
        .groupby(QUESTION_ID_COL, as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    if not buggy_correct[QUESTION_ID_COL].is_unique:
        raise AssertionError("buggy_runs not unique per question_id")

    buggy_correct = buggy_correct[
        [QUESTION_ID_COL, "python_version", "requirements", "buggy_code"]
    ]

    print("buggy_correct missing:")
    print(buggy_correct.isna().sum()[buggy_correct.isna().sum() > 0])

    print("buggy_base missing:")
    print(buggy_base.isna().sum()[buggy_base.isna().sum() > 0])

    final = buggy_base.merge(buggy_correct, on=QUESTION_ID_COL, how="inner")

    if not final[QUESTION_ID_COL].is_unique:
        raise AssertionError(
            "Final patched guidance dataset not unique per question_id"
        )

    missing = final.isna().sum()
    missing = missing[missing > 0]
    print("final missing:")
    print(missing)

    if not missing.empty:
        print(final[final.isna().any(axis=1)].to_string())

    assert final.notnull().all().all()
    return final


# ----------------------------------------------------------------------
# Patched guidance artifacts (PCI / PFR / PSCOT)
# ----------------------------------------------------------------------


def build_patched_code_generation_dataset(
    *,
    df_base: pd.DataFrame,
    pci: pd.DataFrame,
    pfr: pd.DataFrame,
    pscot: pd.DataFrame,
) -> pd.DataFrame:
    pci = _filter_judge_correct(_select_last_attempt(pci, "PCI"), "PCI")
    pfr = _filter_judge_correct(_select_last_attempt(pfr, "PFR"), "PFR")
    pscot = _filter_judge_correct(_select_last_attempt(pscot, "PSCOT"), "PSCOT")

    pci = pci[[QUESTION_ID_COL, "patched_code_intent"]]
    pfr = pfr[[QUESTION_ID_COL, "functional_requirements"]].rename(
        columns={"functional_requirements": "patched_functional_requirements"}
    )
    pscot = pscot[[QUESTION_ID_COL, "patched_scot"]]

    for name, df in {
        "PCI": pci,
        "PFR": pfr,
        "PSCOT": pscot,
        "BASE": df_base,
    }.items():
        if QUESTION_ID_COL not in df.columns:
            raise ValueError(f"[{name}] Missing column: {QUESTION_ID_COL}")
        _assert_unique(df, name)

    df_patched = pci.merge(pfr, on=QUESTION_ID_COL, how="inner").merge(
        pscot, on=QUESTION_ID_COL, how="inner"
    )

    print(f"[MERGE] Patched CI ∩ FR ∩ SCOT = {len(df_patched)}")

    base_cols = [
        "question_id",
        "question_link",
        "question_title",
        "question_body",
        "accepted_answer_body",
        "python_version",
        "requirements",
        "buggy_code",
    ]

    missing = set(base_cols) - set(df_base.columns)
    if missing:
        raise ValueError(f"[BASE] Missing required columns: {missing}")

    df_final = df_base[base_cols].merge(df_patched, on=QUESTION_ID_COL, how="inner")

    print(f"[MERGE] Base ∩ Patched = {len(df_final)}")

    return df_final
