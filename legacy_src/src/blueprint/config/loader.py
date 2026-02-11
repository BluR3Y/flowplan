import json
import os
import glob
import re
from pathlib import Path
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from typing import Dict, Any, Tuple

from blueprint.exceptions import ConfigError
from blueprint.config.schema import SCHEMA

_VAR = re.compile(r"\$\{([^}]+)\}")
APPEND_LIST_KEYS: set[Tuple[str, ...]] = {
    ("sources",), ("compile", "targets"), ("compare", "pairs"), ("export", "workbooks"),
}

def deep_merge(a, b, path: tuple[str, ...] = ()):
    if a is None: return b
    if b is None: return a
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = deep_merge(out.get(k), v, path + (k,))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if path in APPEND_LIST_KEYS:
            return a + b
        return b
    return b

def _resolve_refs(obj, root):
    if isinstance(obj, dict):
        if "$ref" in obj and len(obj) == 1:
            return _resolve_refs(_get_by_path(root, obj["$ref"]), root)
        return {k: _resolve_refs(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_refs(x, root) for x in obj]
    return obj

def _get_by_path(d: dict, dotted: str):
    cur = d
    try:
        for part in dotted.split("."):
            cur = cur[part]
        return cur
    except (KeyError, TypeError):
        raise KeyError(f"$ref not found: {dotted}")

def _interpolate(obj, ctx):
    if isinstance(obj, str):
        def repl(m):
            key = m.group(1)
            # Try config path first, then env var
            try:
                return str(_get_by_path(ctx, key))
            except (KeyError, Exception):
                return os.getenv(key, m.group(0))
        return _VAR.sub(repl, obj)
    if isinstance(obj, list):
        return [_interpolate(x, ctx) for x in obj]
    if isinstance(obj, dict):
        return {k: _interpolate(v, ctx) for k, v in obj.items()}
    return obj

def _expand_includes(base_dir: Path, data: dict, seen: set[str] | None = None) -> dict:
    if seen is None: seen = set()
    includes = data.pop("include", [])
    merged = {}
    
    def _include_one(pattern: str):
        abs_pattern = (base_dir / pattern)
        for p in sorted(glob.glob(str(abs_pattern))):
            ap = str(Path(p).resolve())
            if ap in seen: continue
            seen.add(ap)
            content = json.loads(Path(p).read_text(encoding="utf-8"))
            inc = _expand_includes(Path(p).parent, content, seen)
            nonlocal merged
            merged = deep_merge(merged, inc)

    if isinstance(includes, list):
        for inc in includes: _include_one(inc)
    elif includes:
        _include_one(includes)
        
    return deep_merge(merged, data)

def load_config(entry: str, profile: str | None = None) -> Dict[str, Any]:
    entry_path = Path(entry)
    if not entry_path.exists():
        raise ConfigError(f"Config file not found: {entry}")

    raw = json.loads(entry_path.read_text(encoding="utf-8"))
    base = _expand_includes(entry_path.parent, raw)

    if profile:
        prof_path = entry_path.parent / "profiles" / f"{profile}.json"
        if prof_path.exists():
            prof_data = json.loads(prof_path.read_text(encoding="utf-8"))
            # Expand includes in profile too
            prof_data = _expand_includes(prof_path.parent, prof_data)
            base = deep_merge(base, prof_data)

    resolved = _resolve_refs(base, base)
    resolved = _interpolate(resolved, resolved)

    try:
        Draft202012Validator(SCHEMA).validate(resolved)
    except ValidationError as e:
        loc = " / ".join(str(p) for p in e.path)
        raise ConfigError(f"Config validation error at `{loc or '<root>'}`: {e.message}")

    return resolved