from pathlib import Path
from dataclasses import dataclass

@dataclass
class DataFile:
    model: str
    language: str
    type: str
    cefr: str
    file_name: str
    path: Path