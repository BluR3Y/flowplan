from typing import Any, Union
import operator
import pandas as pd
import numpy as np
from ..exceptions import ExprError

Node = Union[dict, list, int, float, str, bool, None]

_OPS = {
    "==": operator.eq, "!=": operator.ne, ">": operator.gt,
    ">=": operator.ge, "<": operator.lt, "<=": operator.le
}

def _to_series(df: pd.DataFrame, x: Any) -> pd.Series:
    if isinstance(x, pd.Series): return x
    if x is None or isinstance(x, (int, float, str, bool)):
        return pd.Series([x] * len(df), index=df.index)
    raise ExprError(f"Cannot broadcast value of type {type(x)}")

def _as_ast(node: Node) -> dict:
    if isinstance(node, list):
        if not node: raise ExprError("Empty expression array")
        return {"op": str(node[0]), "args": node[1:]}
    if isinstance(node, dict): return node
    return {"op": "lit", "args": [node]}

def build_mask(df: pd.DataFrame, expr: dict) -> pd.Series:
    """Evaluate a filter expression (recursive dict)."""
    if not expr: return pd.Series(True, index=df.index)
    if "AND" in expr:
        return np.logical_and.reduce([build_mask(df, e) for e in expr["AND"]])
    if "OR" in expr:
        return np.logical_or.reduce([build_mask(df, e) for e in expr["OR"]])
    
    # Leaf node: { field: { op: X, value: Y } }
    if len(expr) == 1:
        field, cond = next(iter(expr.items()))
        return _eval_leaf(df, field, cond)
    raise ValueError(f"Invalid filter expression: {expr}")

def _eval_leaf(df: pd.DataFrame, field: str, cond: dict) -> pd.Series:
    if "op" not in cond: raise ValueError("Leaf missing 'op'")
    op = cond["op"]
    val = cond.get("value")

    if op == "in": return df[field].isin(val)
    if op == "not_in": return ~df[field].isin(val)
    if op == "is_null": return df[field].isna()
    if op == "not_null": return ~df[field].isna()
    if op == "between": return (df[field] >= cond.get("start")) & (df[field] <= cond.get("end"))
    if op in _OPS: return _OPS[op](df[field], val)
    raise ValueError(f"Unsupported operator: {op}")

def eval_expr(df: pd.DataFrame, node: Node) -> pd.Series:
    """Evaluate a computational expression AST."""
    ast = _as_ast(node)
    op = ast.get("op")
    args = ast.get("args", [])

    if op == "lit": return _to_series(df, args[0] if args else None)
    if "col" in ast: return df[ast["col"]]

    # Special Lazy evaluation (IF)
    if op == "if":
        cond = eval_expr(df, args[0]).astype("boolean").fillna(False)
        then_val = eval_expr(df, args[1])
        else_val = eval_expr(df, args[2]) if len(args) > 2 else _to_series(df, None)
        return then_val.where(cond, else_val)
    
    # Eager evaluation for others
    ev = [eval_expr(df, a) if isinstance(a, (dict, list)) else _to_series(df, a) for a in args]

    # Arithmetic & Logic
    if op == "add": return sum(ev)
    if op == "sub": return ev[0] - ev[1]
    if op == "mul":
        out = ev[0]
        for x in ev[1:]: out = out * x
        return out
    if op == "div": return ev[0] / ev[1]
    if op == "coalesce":
        out = ev[0]
        for x in ev[1:]: out = out.fillna(x)
        return out
    if op == "concat":
        parts = [s.astype("string") for s in ev]
        out = parts[0]
        for s in parts[1:]: out = out.str.cat(s, na_rep="")
        return out
    
    # Missing Some Operations from original codebase
    # ToDo: Finish This

    raise ExprError(f"Unknown op: {op}")