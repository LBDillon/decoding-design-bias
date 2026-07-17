
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


_PKG = Path(__file__).resolve().parent


def repo_root() -> Path:
    """Return the checkout root, with one optional relocation override."""
    override = os.environ.get("DECODING_BIAS_ROOT")
    return Path(override).expanduser().resolve() if override else _PKG.parents[1]


@dataclass(frozen=True)
class Config:
    root: Path
    analysis_table: Path
    taxonomy_table: Path
    design_dir: Path
    finetune_dir: Path
    pdb_dir: Path
    gam_dir: Path
    expected_dir: Path
    output_dir: Path

    @classmethod
    def load(
        cls,
        *,
        data: str | Path | None = None,
        output_dir: str | Path | None = None,
        root: str | Path | None = None,
        **_: object,
    ) -> "Config":
        base = Path(root).expanduser().resolve() if root else repo_root()

        def resolve(value: str | Path) -> Path:
            path = Path(value).expanduser()
            return path.resolve() if path.is_absolute() else (base / path).resolve()

        return cls(
            root=base,
            analysis_table=resolve(data or "data/main_analysis.csv"),
            taxonomy_table=resolve("data/taxonomy.csv"),
            design_dir=resolve("data/design"),
            finetune_dir=resolve("data/finetune"),
            pdb_dir=resolve("data/pdb"),
            gam_dir=resolve("data/gam"),
            expected_dir=resolve("expected"),
            output_dir=resolve(output_dir or "results"),
        )

    def stage_output(self, name: str) -> Path:
        path = self.output_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path
