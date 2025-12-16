import pandas as pd
import numpy as np
from typing import Any, Union, List, Dict
from flowplan.exceptions import ExprError
from flowplan.utils import load_entrypoints
import operator

# Load plugins for custom expression operators (UDFs)
EXPR_OP_PLUGINS = load_entrypoints("flowplan.expr_ops")

Node = Union[dict, list, int, float, str, bool, None]

def _to_series(df: pd.DataFrame, x: Any) -> pd.Series:
    """
    Broadcast a scalar value to a Series matching the DataFrame's index,
    or return the Series if it is already one.
    """
    if isinstance(x, pd.Series):
        return x
    if x is None or isinstance(x, (int, float, str, bool, np.number)):
        return pd.Series([x] * len(df), index=df.index)
    raise ExprError(f"Cannot broadcast value of type {type(x)}")

def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Retrieve a column from the DataFrame, raising ExprError if missing."""
    if name not in df.columns:
        raise ExprError(f"Unknown column in expression: {name}")
    return df[name]

def _as_ast(node: Node) -> dict:
    """
    Normalize short array syntax to standard AST dict.
    ["add", 1, 2] -> {"op": "add", "args": [1, 2]}
    """
    if isinstance(node, list):
        if not node:
            raise ExprError("Empty expression array")
        op = node[0]
        return {"op": str(op), "args": node[1:]}
    if isinstance(node, dict):
        return node
    # primitives allowed as literals
    return {"op": "lit", "args": [node]}

def _materialize_list(node: Any) -> list:
    """
    Extract a raw list for set membership operations.
    Supports {"list": [...]} or ["item1", "item2"].
    """
    if isinstance(node, dict):
        if "list" in node and isinstance(node["list"], list):
            return node["list"]
        if "set" in node and isinstance(node["set"], list):
            return node["set"]
    if isinstance(node, list):
        return node
    raise ExprError("List membership expects {'list': [...]} (or literal list) as RHS")

def op_in(df: pd.DataFrame, series: pd.Series, values: list) -> pd.Series:
    s = series if isinstance(series, pd.Series) else pd.Series([series]*len(df), index=df.index)
    return s.isin(values)

def op_not_in(df: pd.DataFrame, series: pd.Series, values: list) -> pd.Series:
    s = series if isinstance(series, pd.Series) else pd.Series([series]*len(df), index=df.index)
    return ~s.isin(values)

def eval_expr(df: pd.DataFrame, node: Node) -> pd.Series:
    """
    Recursively evaluate an expression AST against the DataFrame.
    """
    ast = _as_ast(node)
    op = ast.get("op")
    args = ast.get("args", [])

    # --- 1. Literals & References ---
    if op == "lit":
        return _to_series(df, args[0] if args else None)

    if "col" in ast:
        return _col(df, ast["col"])

    # --- 2. Special Forms (Lazy Evaluation or Special Args) ---

    # IF Statement: ["if", cond, then, else]
    # We evaluate condition first, then branches, using .where()
    if op == "if":
        if len(args) < 2:
            raise ExprError("if expects at least 2 args: ['if', cond, then, else?]")
        cond = eval_expr(df, args[0]).astype("boolean").fillna(False)
        then_val = eval_expr(df, args[1]) if len(args) > 1 else _to_series(df, None)
        else_val = eval_expr(df, args[2]) if len(args) > 2 else _to_series(df, None)
        return then_val.where(cond, else_val)
    
    # Membership: ["in", val, list]
    if op in ("in", "not_in"):
        if len(args) != 2:
            raise ExprError(f"{op} expects 2 args: ['{op}', expr, {{'list':[ ... ]}}]")
        left = eval_expr(df, args[0])
        values = _materialize_list(args[1])  # keeps RHS as literal list
        return op_in(df, left, values) if op == "in" else op_not_in(df, left, values)

    # --- 3. Standard Operators (Eager Evaluation) ---

    # Recursively evaluate all arguments
    ev = []
    for a in args:
        if isinstance(a, (dict, list)):
            ev.append(eval_expr(df, a))
        else:
            ev.append(_to_series(df, a))

    # Arithmetic
    if op == "add":
        out = ev[0]
        for s in ev[1:]:
            out = out + s
        return out
    if op == "sub": return ev[0] - ev[1]
    if op == "mul":
        out = ev[0]
        for s in ev[1:]:
            out = out * s
        return out
    if op == "div": return ev[0] / ev[1]
    if op == "pow": return ev[0] ** ev[1]
    if op == "neg": return -ev[0]
    if op == "abs": return ev[0].abs()
    if op == "round":
        nd = int(args[1]) if len(args) > 1 and not isinstance(args[1], (dict, list)) else 0
        return ev[0].round(nd)

    # Comparison (Pandas handles NA propagation)
    if op == "eq":  return ev[0].eq(ev[1])
    if op == "neq": return ev[0].ne(ev[1])
    if op == "gt":  return ev[0] > ev[1]
    if op == "gte": return ev[0] >= ev[1]
    if op == "lt":  return ev[0] < ev[1]
    if op == "lte": return ev[0] <= ev[1]

    # Boolean Logic
    if op == "and":
        out = ev[0].astype("boolean")
        for s in ev[1:]:
            out = out & s.astype("boolean")
        return out
    if op == "or":
        out = ev[0].astype("boolean")
        for s in ev[1:]:
            out = out | s.astype("boolean")
        return out
    if op == "not":
        return (~ev[0].astype("boolean")).astype("boolean")

    # Null Handling
    if op == "coalesce":
        out = ev[0]
        for s in ev[1:]:
            out = out.fillna(s)
        return out
    if op == "fillna":
        return ev[0].fillna(ev[1])
    if op == "is_null":
        return ev[0].isna()
    if op == "not_null":
        return ~ev[0].isna()

    # Strings
    if op == "concat":
        parts = [s.astype("string") for s in ev]
        out = parts[0]
        for s in parts[1:]:
            out = out.str.cat(s, na_rep="")
        return out
    if op == "len":
        return ev[0].astype("string").str.len()

    # Dates
    if op == "strftime":
        fmt = args[0] if isinstance(args[0], str) else "%Y-%m-%d"
        ser = ev[1] if len(ev) > 1 else ev[0]
        ser = pd.to_datetime(ser, errors="coerce")
        return ser.dt.strftime(fmt)
    if op == "datediff":
        unit = (args[0] if isinstance(args[0], str) else "day").lower()
        end = pd.to_datetime(ev[1], errors="coerce")
        start = pd.to_datetime(ev[2], errors="coerce")
        delta = (end - start)
        if unit == "day": return delta.dt.days
        if unit == "hour": return (delta.dt.total_seconds() / 3600)
        if unit == "minute": return (delta.dt.total_seconds() / 60)
        raise ExprError(f"Unsupported datediff unit: {unit}")

    # Misc / Math
    if op == "clip":
        base = ev[0]
        minv = ev[1] if len(ev) > 1 else None
        maxv = ev[2] if len(ev) > 2 else None
        return base.clip(lower=minv if minv is not None else -np.inf,
                         upper=maxv if maxv is not None else  np.inf)
    if op == "percent":
        return ev[0] * ev[1]
    
    # User-Defined Functions (UDFs)
    if op in EXPR_OP_PLUGINS:
        return EXPR_OP_PLUGINS[op](df, *ev)

    raise ExprError(f"Unknown op: {op}")

# -------------------------------------------------------------------------
# Filter Logic
# -------------------------------------------------------------------------

_OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}

def _eval_leaf(df: pd.DataFrame, field: str, cond: Dict[str, Any]) -> pd.Series:
    if "op" in cond:
        op = cond["op"]
        value = cond.get("value")
        if field not in df.columns:
            # If field missing, default to False to avoid crash, or raise error?
            # Raising error is safer for declarative configs.
            raise KeyError(f"Filter field '{field}' not found in dataframe columns: {list(df.columns)}")
            
        col = df[field]
        
        if op == "in":
            return col.isin(value)
        if op == "not_in":
            return ~col.isin(value)
        if op == "is_null":
            return col.isna()
        if op == "not_null":
            return ~col.isna()
        if op == "between":
            start, end = cond.get("start"), cond.get("end")
            return (col >= start) & (col <= end)
        if op in _OPS:
            return _OPS[op](col, value)
        raise ValueError(f"Unsupported operator: {op}")
    raise ValueError("Invalid condition leaf")

def build_mask(df: pd.DataFrame, expr: Dict[str, Any]) -> pd.Series:
    if not expr:
        return pd.Series(True, index=df.index)
    if "AND" in expr:
        masks = [build_mask(df, e) for e in expr["AND"]]
        out = masks[0]
        for m in masks[1:]:
            out = out & m
        return out
    if "OR" in expr:
        masks = [build_mask(df, e) for e in expr["OR"]]
        out = masks[0]
        for m in masks[1:]:
            out = out | m
        return out
    # leaf: { field: { op: X, value: Y } }
    if len(expr) == 1:
        field, cond = next(iter(expr.items()))
        return _eval_leaf(df, field, cond)
    raise ValueError(f"Invalid filter expression: {expr}")