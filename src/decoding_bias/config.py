"""Central configuration + path resolution for the decoding-bias pipeline.

All stages resolve inputs and outputs through a single `Config` instance, so input and
output locations can change without editing source.

Resolution order (lowest to highest precedence):
  1. config_default.yaml (shipped alongside this module)
  2. a user YAML passed via `Config.load(config_path=...)`
  3. environment variables  DECODING_BIAS_<SECTION>_<KEY>
  4. explicit keyword overrides (e.g. CLI --data / --output-dir)

The repository root is `DECODING_BIAS_ROOT` if set, else the parent of `src/`.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_PKG = Path(__file__).resolve().parent           # .../src/decoding_bias
_DEFAULT_YAML = _PKG / "config_default.yaml"


def repo_root() -> Path:
    env = os.environ.get("DECODING_BIAS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return _PKG.parents[1]                        # src/decoding_bias -> src -> repo


def _deep_update(base: dict, extra: dict) -> dict:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _apply_env(cfg: dict) -> None:
    """DECODING_BIAS_PATHS_ANALYSIS_TABLE=... overrides cfg['paths']['analysis_table']."""
    for env_key, val in os.environ.items():
        if not env_key.startswith("DECODING_BIAS_"):
            continue
        rest = env_key[len("DECODING_BIAS_"):].lower()
        if rest == "root":
            continue
        for section in cfg:
            prefix = section + "_"
            if rest.startswith(prefix):
                key = rest[len(prefix):]
                if key in cfg[section]:
                    cfg[section][key] = val
                break


@dataclass
class Config:
    root: Path
    raw: dict

    # ---- construction -----------------------------------------------------
    @classmethod
    def load(cls, config_path: str | Path | None = None, **overrides: Any) -> "Config":
        cfg = yaml.safe_load(_DEFAULT_YAML.read_text())
        if config_path:
            _deep_update(cfg, yaml.safe_load(Path(config_path).read_text()) or {})
        _apply_env(cfg)
        # explicit overrides: data=..., output_dir=... map into paths
        for k in ("analysis_table", "output_dir", "metadata_table",
                  "design_dir", "structures_dir", "weights_dir"):
            if overrides.get(k):
                cfg["paths"][k] = overrides[k]
        if overrides.get("data"):
            cfg["paths"]["analysis_table"] = overrides["data"]
        return cls(root=repo_root(), raw=cfg)

    # ---- path helpers -----------------------------------------------------
    def _resolve(self, value: str) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else (self.root / p)

    @property
    def analysis_table(self) -> Path:
        return self._resolve(self.raw["paths"]["analysis_table"])

    @property
    def output_dir(self) -> Path:
        return self._resolve(self.raw["paths"]["output_dir"])

    def external(self, key: str) -> Path | None:
        """A not-shipped input dir/file (metadata_table, design_dir, …) or None if unset."""
        val = self.raw["paths"].get(key, "")
        return self._resolve(val) if val else None

    def require(self, key: str, what: str) -> Path:
        """Return an external path or raise a clear 'blocked' error naming the config key."""
        p = self.external(key)
        if p is None:
            raise FileNotFoundError(
                f"{what} requires paths.{key}, which is not set (this input is not "
                f"shipped in the repo). Set it in a config YAML or via "
                f"DECODING_BIAS_PATHS_{key.upper()}, then rerun."
            )
        if not p.exists():
            raise FileNotFoundError(f"paths.{key} = {p} does not exist.")
        return p

    def params(self, section: str) -> dict:
        return copy.deepcopy(self.raw.get(section, {}))

    def stage_output(self, name: str) -> Path:
        d = self.output_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d
