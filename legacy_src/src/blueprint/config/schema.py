from typing import Any, Dict

SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["schema", "sources", "compile"],
    "properties": {
        "version": {"type": ["string", "number"]},
        "timezone": {"type": ["string", "null"]},
        "output": {"type": ["string", "null"]},
        "schema": {
            "type": "object",
            "properties": {
                "aliases": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "type": {"enum": ["string", "integer", "number", "date", "boolean"]},
                            "identifier": {"type": "boolean"},
                            "not_null": {"type": "boolean"},
                            "enum": {"type": "array", "items": {"type": "string"}},
                            "date": {
                                "type": "object",
                                "properties": {
                                    "format": {"type": "string"},
                                    "granularity": {"type": "string"},
                                    "abs_tol": {"type": "string"}
                                },
                                "required": ["format"],
                                "additionalProperties": True
                            }
                        },
                        "required": ["type"],
                        "additionalProperties": True
                    }
                }
            },
            "required": ["aliases"]
        },
        "sources": {"type": "array"},
        "compile": {"type": "object"},
        "compare": {"type": "object"},
        "export": {"type": "object"}
    }
}