"""Parameter table for the Vector simulacrum. Authored-by: CPCS.

parameters.json is the single source of every physical number. This module
loads it, checks it, and hands out plain name -> value maps. It needs no
MuJoCo, so the checks run anywhere.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROVENANCE = ("measured", "fitted", "manual", "sdk", "assumed", "chosen")
CHOSEN_KINDS = ("scene", "numeric")
KINDS = ("robot", "scene", "numeric", "reference")
SOURCE_KINDS = ("vendor_firmware", "vendor_webots_model", "vendor_sdk", "household_code")
TABLE_PATH = Path(__file__).resolve().parent.parent / "parameters.json"
MAX_ENTRIES = 512


def load(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read the table and refuse to return one that fails its checks."""
    table_path = Path(path) if path is not None else TABLE_PATH
    with open(table_path, "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    problems = validate(doc)
    if problems:
        raise ValueError("parameter table invalid: " + "; ".join(problems[:8]))
    return doc


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else [value]


def _range_problems(name: str, entry: Dict[str, Any]) -> List[str]:
    """Check low and high bracket the value, component by component."""
    out: List[str] = []
    has_low, has_high = "low" in entry, "high" in entry
    if has_low != has_high:
        return [f"{name}: low and high must come together"]
    if not has_low:
        needs = entry.get("provenance") in ("assumed", "fitted", "chosen")
        return [f"{name}: {entry.get('provenance')} without low/high"] if needs else out
    lows, highs, vals = _as_list(entry["low"]), _as_list(entry["high"]), _as_list(entry["value"])
    if not (len(lows) == len(highs) == len(vals)):
        return [f"{name}: low/high shape differs from value"]
    for low, val, high in zip(lows, vals, highs):
        numeric = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (low, val, high))
        if not numeric or not low <= val <= high:
            out.append(f"{name}: value {val} not inside [{low}, {high}]")
    return out


def _source_problems(name: str, entry: Dict[str, Any], sources: Dict[str, Any]) -> List[str]:
    """A non-assumed entry must name a source that the table defines."""
    out: List[str] = []
    text = entry.get("source", "")
    if not isinstance(text, str) or not text.strip():
        return [f"{name}: empty source"]
    prov = entry.get("provenance")
    if prov == "sdk" and entry.get("source_kind") not in SOURCE_KINDS:
        out.append(f"{name}: sdk entry without a valid source_kind")
    if prov != "sdk" and "source_kind" in entry:
        out.append(f"{name}: source_kind only belongs on sdk entries")
    if prov in ("measured", "fitted") and "REC" not in text:
        out.append(f"{name}: {prov} entry must cite the recording (REC)")
    if (prov == "chosen") != (entry.get("kind") in CHOSEN_KINDS):
        out.append(f"{name}: scene and solver settings are chosen, and nothing about the robot is")
    if prov in ("manual", "sdk"):
        cited = [key for key in sources if text.startswith(key + ":") or text.startswith(key + " ")
                 or ("; " + key + ":") in text or ("; " + key + " ") in text]
        if not cited:
            out.append(f"{name}: source cites no key from the sources block")
        if prov == "manual" and "TRM" not in cited:
            out.append(f"{name}: manual entry must cite TRM")
    return out


def _entry_problems(name: str, entry: Any, sources: Dict[str, Any]) -> List[str]:
    if not isinstance(entry, dict):
        return [f"{name}: entry is not an object"]
    out: List[str] = []
    for key in ("value", "unit", "provenance", "source", "kind"):
        if key not in entry:
            out.append(f"{name}: missing {key}")
    if entry.get("provenance") not in PROVENANCE:
        out.append(f"{name}: provenance {entry.get('provenance')!r} not one of {PROVENANCE}")
    if entry.get("kind") not in KINDS:
        out.append(f"{name}: kind {entry.get('kind')!r} not one of {KINDS}")
    if out:
        return out
    return _range_problems(name, entry) + _source_problems(name, entry, sources)


def validate(doc: Any) -> List[str]:
    """Return every problem found in the table; an empty list means it is sound."""
    if not isinstance(doc, dict) or not isinstance(doc.get("parameters"), dict):
        return ["table has no parameters object"]
    sources = doc.get("sources")
    if not isinstance(sources, dict) or not sources:
        return ["table has no sources block"]
    params = doc["parameters"]
    if not 0 < len(params) <= MAX_ENTRIES:
        return [f"parameter count {len(params)} outside 1..{MAX_ENTRIES}"]
    problems: List[str] = []
    classes = doc.get("provenance_classes")
    if not isinstance(classes, dict) or sorted(classes) != sorted(PROVENANCE):
        problems.append("provenance_classes does not describe exactly the classes the code allows")
    for name in sorted(params):
        problems.extend(_entry_problems(name, params[name], sources))
    for item in doc.get("missing", []):
        if not all(isinstance(item.get(k), str) and item.get(k) for k in ("name", "what", "measure", "needs")):
            problems.append(f"missing-list item {item.get('name')!r} lacks name/what/measure/needs")
    return problems


def values(doc: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Flatten the table to name -> value, with optional overrides by name."""
    assert isinstance(doc, dict) and "parameters" in doc, "need a loaded table"
    out = {name: entry["value"] for name, entry in doc["parameters"].items()}
    for name, value in (overrides or {}).items():
        if name not in out:
            raise KeyError(f"unknown parameter {name!r}")
        out[name] = value
    assert len(out) == len(doc["parameters"]), "override changed the parameter set"
    return out


def sample(doc: Dict[str, Any], seed: int, kinds: Sequence[str] = ("robot",),
           skip: Sequence[str] = ()) -> Dict[str, Any]:
    """Draw every ranged parameter of the given kinds uniformly inside its range.

    By default only robot parameters are drawn; scene, numeric and reference
    entries keep their table value. Names in skip keep their table value too.
    The draw order is the sorted name order, so one seed always gives one result.
    """
    assert isinstance(seed, int), "seed must be an int"
    assert all(kind in KINDS for kind in kinds), "unknown parameter kind"
    rng = random.Random(seed)
    out = values(doc)
    for name in sorted(doc["parameters"]):
        entry = doc["parameters"][name]
        if entry["kind"] not in kinds or "low" not in entry or name in skip:
            continue
        lows, highs = _as_list(entry["low"]), _as_list(entry["high"])
        drawn = [rng.uniform(float(lo), float(hi)) for lo, hi in zip(lows, highs)]
        out[name] = drawn if isinstance(entry["value"], list) else drawn[0]
    assert set(out) == set(doc["parameters"]), "sample changed the parameter set"
    return out


def summary(doc: Dict[str, Any]) -> Dict[str, int]:
    """Count entries by provenance, plus the missing list."""
    assert isinstance(doc, dict) and "parameters" in doc, "need a loaded table"
    counts = {name: 0 for name in PROVENANCE}
    for entry in doc["parameters"].values():
        counts[entry["provenance"]] += 1
    counts["total"] = len(doc["parameters"])
    counts["missing"] = len(doc.get("missing", []))
    assert sum(counts[name] for name in PROVENANCE) == counts["total"], "provenance counts do not add up"
    return counts
