from typing import Dict, List, Any

def filter_missing_keys(obj: Dict[str, Any], required: List[str]) -> List[str]:
    return set(required) - set(obj.keys())