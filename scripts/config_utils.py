import json
from pathlib import Path


def load_json_config(config_path: Path, require_exists: bool = False) -> dict:
    """Load a JSON config file.

    If require_exists=False and the file does not exist, returns {}.
    """
    if not config_path.exists():
        if require_exists:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return {}

    with open(config_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")
    return data


def pick_value(cli_value, config: dict, key: str, default):
    """Resolve a setting with priority: CLI > config file > default."""
    if cli_value is not None:
        return cli_value
    if key in config:
        return config[key]
    return default


def normalize_csv_or_list(value) -> str:
    """Accept either a list or csv-like string and return comma-separated string."""
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    return str(value)
