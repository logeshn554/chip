"""
Design Memory Store — Stores RTL, testbenches, verification metrics, and version lineage.

Uses Git for source version history and structured storage (JSON/SQLite) for:
- RTL source
- testbench
- synthesis results
- functional results
- timing
- area
- power estimate
- reward
- version
- parent design
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class DesignRecord:
    """Full snapshot of a hardware design iteration."""
    design_id: str
    module_name: str
    version: str
    rtl_source: str
    testbench_source: str = ""
    parent_version: str = ""
    functional_results: dict[str, Any] = field(default_factory=dict)
    synthesis_results: dict[str, Any] = field(default_factory=dict)
    area: float = 0.0
    timing: float = 0.0
    power: float = 0.0
    reward: float = 0.0
    git_commit: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DesignStore:
    """Stores design iterations and their associated metrics."""

    def __init__(self, base_dir: str = "./designs"):
        self.base_dir = base_dir
        self.metadata_file = os.path.join(base_dir, "design_catalog.json")
        os.makedirs(base_dir, exist_ok=True)
        self._catalog: dict[str, dict[str, Any]] = self._load_catalog()

    def _load_catalog(self) -> dict[str, dict[str, Any]]:
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load design catalog: {e}")
        return {}

    def _save_catalog(self) -> None:
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self._catalog, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save design catalog: {e}")

    def save_design(self, record: DesignRecord) -> str:
        """Save a new design version and persist its files."""
        # Save RTL file to disk
        design_dir = os.path.join(self.base_dir, record.module_name, record.version)
        os.makedirs(design_dir, exist_ok=True)

        rtl_path = os.path.join(design_dir, f"{record.module_name}.sv")
        with open(rtl_path, "w", encoding="utf-8") as f:
            f.write(record.rtl_source)

        if record.testbench_source:
            tb_path = os.path.join(design_dir, f"test_{record.module_name}.py")
            with open(tb_path, "w", encoding="utf-8") as f:
                f.write(record.testbench_source)

        key = f"{record.module_name}@{record.version}"
        self._catalog[key] = asdict(record)
        self._save_catalog()
        logger.info(f"Saved design: {key} (Reward: {record.reward})")
        return key

    def get_design(self, module_name: str, version: Optional[str] = None) -> Optional[DesignRecord]:
        """Retrieve a specific version or latest version."""
        if version:
            key = f"{module_name}@{version}"
            data = self._catalog.get(key)
            return DesignRecord(**data) if data else None

        # Find latest by timestamp
        matching = [
            DesignRecord(**d) for d in self._catalog.values()
            if d.get("module_name") == module_name
        ]
        if not matching:
            return None
        matching.sort(key=lambda x: x.timestamp, reverse=True)
        return matching[0]

    def compare_designs(self, module_name: str, ver_a: str, ver_b: str) -> dict[str, Any]:
        """Compare metrics and source between two design versions."""
        a = self.get_design(module_name, ver_a)
        b = self.get_design(module_name, ver_b)
        if not a or not b:
            return {"error": f"One or both versions not found: {ver_a}, {ver_b}"}

        return {
            "module": module_name,
            "version_a": ver_a,
            "version_b": ver_b,
            "reward_delta": b.reward - a.reward,
            "area_delta": b.area - a.area,
            "functional_pass_a": a.functional_results.get("pass_rate", 0.0),
            "functional_pass_b": b.functional_results.get("pass_rate", 0.0),
            "source_diff_bytes": len(b.rtl_source) - len(a.rtl_source),
        }
