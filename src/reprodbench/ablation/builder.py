from .context import BuggySemanticContext, PatchedSemanticContext
from .mode import AblationMode


def build_buggy_semantic_context(
    *,
    ablation: AblationMode,
    ci: str | None,
    fr: str | None,
    scot: str | None,
) -> BuggySemanticContext:

    return BuggySemanticContext(
        buggy_code_intent=(
            ci
            if ablation
            in {
                AblationMode.CI,
                AblationMode.CI_FR,
                AblationMode.CI_SCOT,
                AblationMode.CI_FR_SCOT,
            }
            else None
        ),
        buggy_functional_requirements=(
            fr
            if ablation
            in {
                AblationMode.FR,
                AblationMode.CI_FR,
                AblationMode.FR_SCOT,
                AblationMode.CI_FR_SCOT,
            }
            else None
        ),
        buggy_scot=(
            scot
            if ablation
            in {
                AblationMode.SCOT,
                AblationMode.CI_SCOT,
                AblationMode.FR_SCOT,
                AblationMode.CI_FR_SCOT,
            }
            else None
        ),
    )


def build_patched_semantic_context(
    *,
    ablation: AblationMode,
    ci: str | None,
    fr: str | None,
    scot: str | None,
) -> PatchedSemanticContext:

    return PatchedSemanticContext(
        patched_code_intent=(
            ci
            if ablation
            in {
                AblationMode.CI,
                AblationMode.CI_FR,
                AblationMode.CI_SCOT,
                AblationMode.CI_FR_SCOT,
            }
            else ""
        ),
        patched_functional_requirements=(
            fr
            if ablation
            in {
                AblationMode.FR,
                AblationMode.CI_FR,
                AblationMode.FR_SCOT,
                AblationMode.CI_FR_SCOT,
            }
            else ""
        ),
        patched_scot=(
            scot
            if ablation
            in {
                AblationMode.SCOT,
                AblationMode.CI_SCOT,
                AblationMode.FR_SCOT,
                AblationMode.CI_FR_SCOT,
            }
            else ""
        ),
    )
