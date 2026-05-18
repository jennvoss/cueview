import yaml


def load_config(path: str = "config/config.yaml") -> dict:
    """Load YAML config. Raise FileNotFoundError if missing."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def override_config(cfg: dict, overrides: dict) -> dict:
    """Deep-merge CLI overrides into cfg. Return merged dict. Do not mutate inputs."""
    result = cfg.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = {**result[key], **value}
        else:
            result[key] = value
    return result
