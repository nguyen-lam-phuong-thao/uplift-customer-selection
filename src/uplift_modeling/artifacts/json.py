"""JSON artifact persistence helpers."""

import json
from pathlib import Path
from typing import Any


def save_json_artifact(payload: dict[str, Any], output_path: Path) -> Path:
    """Save a dictionary as a formatted JSON artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    return output_path
