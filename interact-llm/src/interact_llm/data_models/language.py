from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Language:
    key: str 
    names: Dict[str, str]


supported_languages = {
    "english": Language(
        key = "english",
        names = {
            "english": "English",
            "native": "English",
        },
    ),
    "lithuanian": Language(
        key = "lithuanian",
        names = {
            "english": "Lithuanian",
            "native": "Lietuvių",
        },
    ),
}
