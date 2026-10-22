"""YAML -> nested namespace, with `--set a.b=c` style overrides."""
from __future__ import annotations

from types import SimpleNamespace

import yaml


def _to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
    return d


def to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        return {k: to_dict(v) for k, v in vars(ns).items()}
    return ns


def _set(d: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def load_config(path: str, overrides: list[str] | None = None) -> SimpleNamespace:
    with open(path) as f:
        raw = yaml.safe_load(f)
    for item in overrides or []:
        key, val = item.split("=", 1)
        _set(raw, key, yaml.safe_load(val))
    return _to_ns(raw)


def from_dict(d: dict) -> SimpleNamespace:
    """Rebuild a config namespace from the dict stored inside a checkpoint."""
    return _to_ns(d)
