import pytest
import json
from jsonschema import validate, ValidationError
from flowplan.config.schema import SCHEMA

# Helper to validate a dict against the schema
def validate_config(data):
    validate(instance=data, schema=SCHEMA)

def test_minimal_valid_config():
    """
    Ensures the absolute minimum config passes validation.
    """
    data = {
        "version": "1.0",
        "schema": { "aliases": {} },
        "sources": [],
        "compile": { "targets": [] },
        "output": "./out"
    }
    # Should not raise
    validate_config(data)

def test_schema_aliases_valid():
    """
    Tests valid schema alias definitions.
    """
    data = {
        "version": "1.0",
        "output": "./out",
        "compile": {},
        "sources": [],
        "schema": {
            "aliases": {
                "id": { "type": "integer", "identifier": True },
                "status": { "type": "string", "enum": ["A", "B"] },
                "active": { "type": "boolean", "not_null": True },
                "joined": { "type": "date", "date": { "format": "%Y-%m-%d" } }
            }
        }
    }
    validate_config(data)

def test_schema_aliases_invalid_type():
    """
    Tests that an invalid type raises a ValidationError.
    """
    data = {
        "version": "1.0",
        "output": "./out",
        "compile": {},
        "sources": [],
        "schema": {
            "aliases": {
                "bad_col": { "type": "complex_matrix" } # Invalid type
            }
        }
    }
    with pytest.raises(ValidationError) as exc:
        validate_config(data)
    assert "'complex_matrix' is not one of" in str(exc.value)

def test_sources_structure():
    """
    Tests the structure of sources (Excel vs Inline).
    """
    data = {
        "version": "1.0",
        "output": "./out",
        "schema": { "aliases": {} },
        "compile": {},
        "sources": [
            {
                "id": "excel_src",
                "path": "data.xlsx",
                "tables": [
                    {
                        "name": "Sheet1",
                        "columns": {
                            "Raw Col": { "alias": "clean_col" }
                        }
                    }
                ]
            },
            {
                "id": "inline_src",
                "data": [
                    {
                        "table_id": "t1",
                        "orient": "records",
                        "rows": []
                    }
                ]
            }
        ]
    }
    validate_config(data)

def test_enrichment_structure():
    """
    Tests the complex structure of enrichment blocks inside compile.
    """
    data = {
        "version": "1.0",
        "output": "./out",
        "schema": { "aliases": {} },
        "sources": [],
        "compile": {
            "targets": [
                {
                    "name": "enriched_target",
                    "inputs": ["src"],
                    "enrich": [
                        {
                            "from": "lookup",
                            "left_on": "k1",
                            "right_on": "k2",
                            "add": { "new_col": "ref_col" },
                            "match": {
                                "strategy": ["fuzzy"],
                                "fuzzy": { "threshold": 90, "scorer": "ratio" }
                            }
                        }
                    ]
                }
            ]
        }
    }
    # Note: If your schema definition (SCHEMA dict) is loose on 'compile', 
    # this might pass easily. If strict, it validates the structure.
    validate_config(data)

def test_missing_required_fields():
    """
    Tests that missing root-level required keys raise errors.
    """
    # Missing 'sources'
    data = {
        "version": "1.0",
        "output": "./out",
        "schema": { "aliases": {} },
        "compile": {}
    }
    with pytest.raises(ValidationError) as exc:
        validate_config(data)
    assert "'sources' is a required property" in str(exc.value)