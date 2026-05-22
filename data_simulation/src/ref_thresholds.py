def check_thresholds(
    metrics: dict,
    lower_mdd: float,
    upper_mdd: float,
    lower_len: float,
    upper_len: float,
) -> list:
    """
    Checks whether the evaluated metrics fall within the baseline thresholds.
    """
    mdd = metrics["mdd_mean"]
    length = metrics["avg_text_length"]

    issues = []

    if mdd < lower_mdd:
        issues.append("low")
    elif mdd > upper_mdd:
        issues.append("high")
    if length < lower_len:
        issues.append("short")
    elif length > upper_len:
        issues.append("long")

    return issues