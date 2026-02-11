from typing import Dict, Callable
from importlib.metadata import entry_points

import logging

log = logging.getLogger(__name__)

def load_entrypoints(group: str) -> Dict[str, Callable]:
    """
    Load plugins using importlib.metadata
    """
    try:
        eps = entry_points()
        # Handle Python 3.10+ select() vs older dictionary interface
        if hasattr(eps, "select"):
            candidates = eps.select(group=group)
        else:
            candidates = eps.get(group, [])
        return {ep.name: ep.load() for ep in candidates}
    except Exception as e:
        log.warning(f"Failed to load entry points for {group}: {e}")
        return {}