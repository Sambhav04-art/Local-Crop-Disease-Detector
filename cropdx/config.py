from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    p = Path(path) if path else ROOT / "config.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def path_of(cfg, key, default=None):
    p = Path(cfg["paths"].get(key, default))
    return p if p.is_absolute() else ROOT / p
