import json
import os
import re
import glob
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .exceptions import ConfigError

log = logging.getLogger(__name__)

_VAR_RE = re.compile(r"\$\{([^}]+)\}")
APPEND_LIST_KEYS: Set[Tuple[str, ...]] = {
    ("sources",),
    ("compile", "targets"),
    ("compare", "pairs"),
    ("export", "targets"),
}

_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["schema", "sources", "compile"],
    "properties": {
        "version": {"type": ["string", "number"]},
        "timezone": {"type": ["string", "null"]},
        "output": {"type": ["string", "null"]},
        "schema": {"type": "object", "required": ["aliases"]},
        "sources": {"type": "array"},
        "compile": {"type": "object"},
        "compare": {"type": "object"},
        "export": {
            "type": "object",
            "properties": {
                "targets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["type"], # 'type' is now required (e.g. 'excel', 'sql')
                    }
                }
            }
        }
    }
}

@dataclass
class Config:
    raw: Dict[str, Any]

    @property
    def version(self) -> str:
        return str(self.raw.get("version", "1"))
    
    @property
    def output_dir(self) -> Optional[str]:
        return self.raw.get("output")
    
    @property
    def schema_aliases(self) -> Dict[str, Any]:
        return self.raw.get("schema", {}).get("aliases", {})
    
    @property
    def sources(self) -> List[Dict[str, Any]]:
        return self.raw.get("sources", [])
    
    @property
    def compile_targets(self) -> List[Dict[str, Any]]:
        return self.raw.get("compile", {}).get("targets", [])
    
    @property
    def compare_pairs(self) -> List[Dict[str, Any]]:
        return self.raw.get("compare", {}).get("pairs", [])
    
    @property
    def export_targets(self) -> List[Dict[str, Any]]:
        return self.raw.get("export", {}).get("targets", [])
    
def _deep_merge(a: Any, b: Any, path: Tuple[str, ...] = ()) -> Any:
    if a is None: return b
    if b is None: return a
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = _deep_merge(out.get(k), v, path + (k,))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if path in APPEND_LIST_KEYS:
            return a + b
        return b
    return b

def _expand_includes(base_dir: Path, data: dict, seen: Set[str] | None = None) -> dict:
    if seen is None: seen = set()
    includes = data.pop("include", [])
    merged: dict = {}

    def _include_one(pattern: str):
        abs_pattern = (base_dir / pattern)
        for p in sorted(glob.glob(str(abs_pattern))):
            ap = str(Path(p).resolve())
            if ap in seen: continue
            seen.add(ap)
            inc = _load_json_with_includes(Path(p), seen)
            nonlocal merged
            merged = _deep_merge(merged, inc)
    
    if isinstance(includes, list):
        for inc in includes: _include_one(inc)
    elif includes:
        _include_one(includes)
    
    return _deep_merge(merged, data)

def _load_json_with_includes(path: Path, seen: Set[str] | None = None) -> dict:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return _expand_includes(path.parent, raw, seen)

def _get_by_path(d: dict, dotted: str) -> Any:
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"$ref not found: {dotted}")
        cur = cur[part]
    return cur

def _resolve_refs(obj: Any, root: dict) -> Any:
    if isinstance(obj, dict):
        if "$ref" in obj and len(obj) == 1:
            return _resolve_refs(_get_by_path(root, obj["$ref"]), root)
        return {k: _resolve_refs(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_refs(x, root) for x in obj]
    return obj

def _interp(value: str, ctx: dict) -> str:
    def repl(m):
        key = m.group(1)
        if "." in key:
            try: return str(_get_by_path(ctx, key))
            except Exception: pass
        return os.getenv(key, m.group(0))
    return _VAR_RE.sub(repl, value)

def _interpolate(obj: Any, ctx: dict) -> Any:
    if isinstance(obj, str): return _interp(obj, ctx)
    if isinstance(obj, list): return [_interpolate(x, ctx) for x in obj]
    if isinstance(obj, dict): return {k: _interpolate(v, ctx) for k, v in obj.items()}
    return obj

def load_config(entry_path: str, profile: str | None = None) -> Config:
    entry = Path(entry_path)
    base = _load_json_with_includes(entry)

    if profile:
        prof_path = entry.parent / "profiles" / f"{profile}.json"
        if prof_path.exists():
            base = _deep_merge(base, _load_json_with_includes(prof_path))
    
    resolved = _resolve_refs(base, base)
    resolved = _interpolate(resolved, resolved)

    try:
        Draft202012Validator(_SCHEMA).validate(resolved)
    except ValidationError as e:
        loc = " / ".join(str(p) for p in e.path)
        raise ConfigError(f"Config validation error at `{loc or '<root>'}`: {e.message}")
    
    return Config(resolved)