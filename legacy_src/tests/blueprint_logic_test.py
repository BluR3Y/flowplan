import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from blueprint.config.validator import enforce_types, validate_frame
from blueprint.engine.expressions import eval_expr, build_mask
from blueprint.ops.transforms import apply_pipeline
from blueprint.ops.fuzzy import fuzzy_match_series
from blueprint.ops.enrich import enrich_join
from blueprint.core.pipeline import Pipeline
from blueprint.core.models import Config
from blueprint.exceptions import SourceError, ExprError

# -------------------------------------------------------------------------
# 1. Schema Validation Tests
# -------------------------------------------------------------------------

def test_enforce_types_coercion():
    """
    Tests that string inputs ("True", "0", "1") are correctly coerced to
    boolean, integer, and number types using the validator logic.
    """
    df = pd.DataFrame({
        "id": ["1", "2", "3.0", "bad"],
        "active": ["True", "False", "true", "0"],
        "score": ["10.5", "20", "NaN", "0"],
        "date": ["2023-01-01", "2023/01/02", "bad_date", None]
    })
    
    aliases = {
        "id": {"type": "integer"},
        "active": {"type": "boolean"},
        "score": {"type": "number"},
        "date": {"type": "date", "date": {"format": "%Y-%m-%d"}}
    }
    
    out = enforce_types(df, aliases)
    
    # Integers
    assert pd.api.types.is_integer_dtype(out["id"])
    assert out["id"].iloc[0] == 1
    assert pd.isna(out["id"].iloc[3])  # "bad" -> NaN

    # Booleans (Fixing the previous is True assertion error)
    # Pandas nullable boolean array can return np.True_ or python True
    assert pd.api.types.is_bool_dtype(out["active"])
    assert out["active"].iloc[0] == True  # "True"
    assert out["active"].iloc[1] == False # "False"
    assert out["active"].iloc[2] == True  # "true"
    assert out["active"].iloc[3] == False # "0"
    
    # Floats
    assert pd.api.types.is_float_dtype(out["score"])
    assert out["score"].iloc[0] == 10.5
    
    # Dates
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    assert out["date"].iloc[0] == pd.Timestamp("2023-01-01")
    assert pd.isna(out["date"].iloc[2]) # "bad_date" -> NaT

def test_validate_frame_constraints():
    """Tests constraints: identifier uniqueness, enum validity, not_null."""
    df = pd.DataFrame({
        "id": [1, 2, 2],         # Duplicate ID
        "status": ["A", "B", "C"], # "C" not in enum
        "required": [1, None, 3] # Null in not_null
    })
    
    aliases = {
        "id": {"type": "integer", "identifier": True},
        "status": {"type": "string", "enum": ["A", "B"]},
        "required": {"type": "integer", "not_null": True}
    }
    
    with pytest.raises(SourceError) as exc:
        validate_frame(df, aliases)
    
    msg = str(exc.value)
    assert "identifier must be unique" in msg
    assert "values outside enum" in msg
    assert "contains nulls but not_null=true" in msg

# -------------------------------------------------------------------------
# 2. Expression Engine (AST) Tests
# -------------------------------------------------------------------------

def test_eval_expr_arithmetic_logic():
    df = pd.DataFrame({
        "a": [10, 20, 30],
        "b": [2, 4, 0],
        "status": ["Active", "Inactive", "Active"]
    })
    
    # Math: (a * 2 + 5)
    expr_math = ["add", ["mul", {"col": "a"}, 2], 5]
    res_math = eval_expr(df, expr_math)
    assert res_math.tolist() == [25, 45, 65]
    
    # Logic: if status == Active then a else b
    expr_if = ["if", ["eq", {"col": "status"}, "Active"], {"col": "a"}, {"col": "b"}]
    res_if = eval_expr(df, expr_if)
    assert res_if.tolist() == [10, 4, 30]

def test_eval_expr_dates_strings():
    df = pd.DataFrame({
        "d1": pd.to_datetime(["2023-01-01", "2023-01-01"]),
        "d2": pd.to_datetime(["2023-01-02", "2023-01-04"]),
        "s": ["foo", "bar"]
    })
    
    # Date Diff
    res_diff = eval_expr(df, ["datediff", "day", {"col": "d2"}, {"col": "d1"}])
    assert res_diff.tolist() == [1, 3]
    
    # String Concat
    res_cat = eval_expr(df, ["concat", {"col": "s"}, "_suffix"])
    assert res_cat.tolist() == ["foo_suffix", "bar_suffix"]

def test_build_mask_filtering():
    df = pd.DataFrame({
        "val": [10, 20, 30, 40],
        "cat": ["A", "A", "B", "B"]
    })
    
    # Filter: val > 15 AND cat == 'A'
    expr = {
        "AND": [
            {"val": {"op": ">", "value": 15}},
            {"cat": {"op": "==", "value": "A"}}
        ]
    }
    
    mask = build_mask(df, expr)
    assert mask.tolist() == [False, True, False, False]

# -------------------------------------------------------------------------
# 3. Fuzzy Matching Tests
# -------------------------------------------------------------------------

def test_fuzzy_match_series_optimization():
    """
    Tests the fuzzy matching logic, ensuring normalized matches work.
    """
    # Left side has messy data
    left = pd.Series(["Google", "Google.", "Microsoft", "Apple", "Unknown"])
    # Right side canonical list
    right_unique = pd.Series(["Google Inc", "Microsoft Corp", "Apple Inc"])
    
    matched_vals, scores, methods = fuzzy_match_series(
        left=left,
        right_unique=right_unique,
        normalize_steps=["strip", "lower", "strip_punct"],
        scorer="token_set_ratio", # Very forgiving scorer
        threshold=70,             # Lower threshold for safety in test
        top_k=1,
        block="first_char"
    )
    
    # 1. "Google" -> "Google Inc"
    assert matched_vals[0] == "Google Inc"
    assert scores[0] >= 70
    
    # 2. "Google." -> "Google Inc" (Normalization handles the dot)
    assert matched_vals[1] == "Google Inc"
    
    # 3. "Microsoft" -> "Microsoft Corp"
    assert matched_vals[2] == "Microsoft Corp"
    
    # 4. "Apple" -> "Apple Inc"
    assert matched_vals[3] == "Apple Inc"
    
    # 5. "Unknown" -> Miss
    # Depending on implementation, this might be NaN or None
    assert pd.isna(matched_vals[4])

# -------------------------------------------------------------------------
# 4. Transform Pipeline Tests
# -------------------------------------------------------------------------

def test_transforms_chain():
    s = pd.Series(["  Phone: 123-456  ", "Phone: 555-999"])
    
    steps = [
        {"regex_replace": {"pattern": "Phone: ", "repl": ""}}, 
        {"normalize": ["strip"]},                              
        {"affix": {"text": "+1-", "position": "prefix"}}       
    ]
    
    out = apply_pipeline(s, steps)
    assert out[0] == "+1-123-456"
    assert out[1] == "+1-555-999"

# -------------------------------------------------------------------------
# 5. Enrichment Logic Tests
# -------------------------------------------------------------------------

def test_enrich_join_logic():
    """
    Tests joining two dataframes using approximate matching.
    """
    # Left: Transaction table with messy names
    left = pd.DataFrame({
        "trans_id": [1, 2, 3],
        "client_raw": ["Acme Corp", "Acme", "Unknown LLC"]
    })
    
    # Right: Master client list
    right = pd.DataFrame({
        "client_clean": ["Acme Corporation", "Zeta Inc"],
        "client_id": [100, 200]
    })
    
    add_map = {"final_id": "client_id"}
    match_cfg = {
        "strategy": ["exact", "fuzzy"],
        "fuzzy": {"threshold": 60}, # Low threshold to guarantee 'Acme' matches 'Acme Corporation'
        "on_miss": "leave_null"
    }
    
    out = enrich_join(
        left, right,
        left_on="client_raw",
        right_on="client_clean",
        add=add_map,
        match=match_cfg
    )
    
    # Assertions using safe NA checks
    
    # 1. "Acme Corp" should match "Acme Corporation"
    # Using token_set_ratio this is usually 100, but we use safe logic.
    val0 = out["final_id"][0]
    assert pd.notna(val0), f"Expected match for Acme Corp, got NA"
    assert val0 == 100
    
    # 2. "Acme" should match "Acme Corporation"
    val1 = out["final_id"][1]
    assert pd.notna(val1), f"Expected match for Acme, got NA"
    assert val1 == 100
    
    # 3. "Unknown LLC" should be NA
    val2 = out["final_id"][2]
    assert pd.isna(val2), f"Expected NA for Unknown LLC, got {val2}"

# -------------------------------------------------------------------------
# 6. Integration Test: Pipeline Run
# -------------------------------------------------------------------------

def test_pipeline_union(tmp_path):
    """
    Simulates a full run: Sources -> Compile (Union) -> check DataFrame.
    """
    config_dict = {
        "version": 1.0,
        "output": str(tmp_path),
        "schema": {
            "aliases": {
                "id": {"type": "integer", "identifier": True},
                "amt": {"type": "number"}
            }
        },
        "sources": [
            {
                "data": [
                    {"table_id": "t1", "rows": [{"id": 1, "amt": 10}, {"id": 2, "amt": 20}]},
                    {"table_id": "t2", "rows": [{"id": 3, "amt": 30}]}
                ]
            }
        ],
        "compile": {
            "targets": [
                {
                    "name": "master",
                    "mode": "union",
                    "inputs": ["t1", "t2"]
                }
            ]
        },
        "export": {"workbooks": []}
    }
    
    cfg = Config(config_dict)
    pipe = Pipeline(cfg)
    
    pipe.run()
    
    # Check that 'master' dataframe was created in memory
    assert "master" in pipe.compiled
    master = pipe.compiled["master"]
    
    assert len(master) == 3
    # Check IDs sorted or present
    ids = sorted(master["id"].tolist())
    assert ids == [1, 2, 3]