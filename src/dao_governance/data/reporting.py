"""JSON-safe reporting of explicitly undefined numeric metrics."""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping

import numpy as np


def json_safe_report(report: Mapping[str, Any]) -> Dict[str, Any]:
    undefined = []

    def convert(value: Any, path: str) -> Any:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            undefined.append(path)
            return None
        if isinstance(value, dict):
            return {key: convert(item, f"{path}.{key}") for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item, f"{path}[{index}]") for index, item in enumerate(value)]
        return value

    result = convert(dict(report), "report")
    if "undefined_metrics" in result:
        raise ValueError("The report reserves the undefined_metrics field.")
    result["undefined_metrics"] = undefined
    return result
