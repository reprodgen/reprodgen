from enum import Enum, auto


class AblationMode(Enum):
    NONE = auto()
    CI = auto()
    FR = auto()
    SCOT = auto()
    CI_FR = auto()
    CI_SCOT = auto()
    FR_SCOT = auto()
    CI_FR_SCOT = auto()
