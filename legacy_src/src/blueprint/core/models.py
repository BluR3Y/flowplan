from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class Config:
    raw: Dict[str, Any]

    @property
    def version(self) -> str:
        return str(self.raw.get("version", "1"))
    
    @property
    def timezone(self) -> Optional[str]:
        return self.raw.get("timezone")
    
    @property
    def output(self) -> Optional[str]:
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
    def export_workbooks(self) -> List[Dict[str, Any]]:
        return self.raw.get("export", {}).get("workbooks", [])