import stanza
from data_analysis.src.logger import logger


def get_stanza(processors: str = "tokenize,pos,lemma"):
    """
    Initiates Stanza for Lithuanian text processing.
    """
    logger.info("Downloading Lithuanian Stanza model.")
    stanza.download("lt")

    logger.info("Initializing Stanza processor.")
    processor = stanza.Pipeline(
        lang="lt", processors=processors, tokenize_pretokenized=False
    )
    return processor

def parse_features(feats_str: str) -> dict:
    """
    Parses a UD morphological feature string into a key-value dictionary.

    Example:
        "Mood=Ind|Number=Sing|VerbForm=Fin" -> {"Mood": "Ind", "Number": "Sing", "VerbForm": "Fin"}
        "_" -> {}
    """
    if not feats_str or feats_str == "_":
        return {}
    return dict(item.split("=", 1) for item in feats_str.split("|") if "=" in item)


def finite_verb(feats: dict) -> bool:
    """Finite verbs (VerbForm=Fin): inflected for tense, mood, and person."""
    return feats.get("VerbForm") == "Fin"


def participle(feats: dict) -> bool:
    """
    Any participial form (VerbForm=Part).
    Includes half-participles and active/passive present participles as subsets.
    """
    return feats.get("VerbForm") == "Part"


def adverbial_participle(feats: dict) -> bool:
    """
    Non-finite adverbial verbal forms: converbs (VerbForm=Conv),
    gerunds (VerbForm=Ger), and gerundives (VerbForm=Gdv).
    """
    return feats.get("VerbForm") in {"Conv", "Ger", "Gdv"}


def half_participle(feats: dict) -> bool:
    """
    Lithuanian half-participle (pusdalyvis).
    Defined as an active, present-tense, nominative participle (VerbForm=Part,
    Tense=Pres, Case=Nom, Voice=Act). A subset of participle().
    """
    return (
        feats.get("VerbForm") == "Part"
        and feats.get("Tense") == "Pres"
        and feats.get("Case") == "Nom"
        and feats.get("Voice") == "Act"
    )


def active_present_participle(feats: dict) -> bool:
    """
    Active voice present participle (veikiamosios rūšies esamojo laiko dalyvis).
    Defined as VerbForm=Part, Tense=Pres, Voice=Act.
    Note: includes half-participles, which are a nominative subset of this category.
    """
    return (
        feats.get("VerbForm") == "Part"
        and feats.get("Tense") == "Pres"
        and feats.get("Voice") == "Act"
    )


def passive_present_participle(feats: dict) -> bool:
    """
    Passive voice present participle (neveikiamosios rūšies esamojo laiko dalyvis).
    Defined as VerbForm=Part, Tense=Pres, Voice=Pass.
    """
    return (
        feats.get("VerbForm") == "Part"
        and feats.get("Tense") == "Pres"
        and feats.get("Voice") == "Pass"
    )
