import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Mapping, Tuple, List
from pathlib import Path

@dataclass
class CompareOptions:
    id_label: str = "Baseline"
    new_label: str = "Current"
    include_unchanged_in_bycol: bool = False
    show_delta_cols: bool = True
    # Conditional formatting thresholds for numbers
    delta_iconset: bool = True
    number_precision: int = 6  # rounding for numeric equality
    # How to handle duplicate keys across key_cols
    duplicate_key_mode: str = "pair_by_order"  # [pair_by_order|error]
    # Sheet names
    readme_name: str = "README & Legend"
    summary_name: str = "Summary"
    changes_name: str = "Changes"
    new_rows_name: str = "New Rows"
    missing_rows_name: str = "Missing Rows"


def _safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def _align_keys(df: pd.DataFrame, key_cols: Sequence[str]) -> pd.MultiIndex:
    """Return a MultiIndex built from key_cols (object-typed)."""
    if not key_cols:
        raise ValueError("key_cols must be non-empty")
    if isinstance(key_cols, str):
        key_cols = [key_cols]
    for k in key_cols:
        if k not in df.columns:
            raise KeyError(f"Key column missing: {k}")
    return pd.MultiIndex.from_frame(df[key_cols].astype(object))


def _coerce_numeric(s: pd.Series) -> pd.Series:
    s2 = pd.to_numeric(s, errors="coerce")
    return s2


def _equal_values(a: pd.Series, b: pd.Series, precision: int) -> pd.Series:
    # Try numeric compare with tolerance first
    an = _coerce_numeric(a)
    bn = _coerce_numeric(b)
    both_num = an.notna() & bn.notna()
    equal_num = (an.round(precision) == bn.round(precision))
    # Fallback to string compare for non-numeric or NaNs
    ae = a.astype("string").fillna(pd.NA)
    be = b.astype("string").fillna(pd.NA)
    equal_str = (ae == be)
    return np.where(both_num, equal_num, equal_str)


def _prep_dfs(
    df_left: pd.DataFrame, df_right: pd.DataFrame, key_cols: Sequence[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare left/right frames for diffing with a guaranteed-unique MultiIndex.
    Always appends a per-key sequence level (via cumcount) so duplicates are
    uniquely indexed.
    """
    left = df_left.copy()
    right = df_right.copy()

    # Validate key columns exist
    if isinstance(key_cols, str):
        key_cols = [key_cols]
    for k in key_cols:
        if k not in left.columns or k not in right.columns:
            raise KeyError(f"Key column missing in one of the frames: {k}")

    # Stable sort for deterministic pairing
    left_sorted = left.sort_values(list(key_cols), kind="mergesort")
    right_sorted = right.sort_values(list(key_cols), kind="mergesort")

    # Per-key sequence number (handles duplicates AND NaN keys)
    left_sorted["__row__"] = left_sorted.groupby(list(key_cols), dropna=False).cumcount()
    right_sorted["__row__"] = right_sorted.groupby(list(key_cols), dropna=False).cumcount()

    # Build unique MultiIndex = key_cols + ["__row__"]
    left_index = pd.MultiIndex.from_frame(
        left_sorted[list(key_cols) + ["__row__"]].astype(object)
    )
    right_index = pd.MultiIndex.from_frame(
        right_sorted[list(key_cols) + ["__row__"]].astype(object)
    )

    left = left_sorted.set_index(left_index)
    right = right_sorted.set_index(right_index)

    # Safety net: ensure uniqueness (paranoid check)
    if not left.index.is_unique:
        left = left.reset_index(drop=False)
        left["__i__"] = np.arange(len(left))
        # rebuild from displayed key levels + __row__ + unique counter
        mi = pd.MultiIndex.from_frame(
            left[key_cols + ["__row__", "__i__"]].astype(object)
        )
        left = left.set_index(mi)
        
    if not right.index.is_unique:
        right = right.reset_index(drop=False)
        right["__i__"] = np.arange(len(right))
        mi = pd.MultiIndex.from_frame(
            right[key_cols + ["__row__", "__i__"]].astype(object)
        )
        right = right.set_index(mi)

    # unify columns
    all_cols = list(dict.fromkeys([*left.columns.tolist(), *right.columns.tolist()]))
    left = left.reindex(columns=all_cols)
    right = right.reindex(columns=all_cols)

    # union of keys and align
    union_index = left.index.union(right.index)
    left = left.reindex(union_index)
    right = right.reindex(union_index)
    return left, right, pd.DataFrame(index=union_index)


def _compute_status(
    left: pd.DataFrame,
    right: pd.DataFrame,
    compare_cols: Sequence[str],
    precision: int,
) -> Tuple[pd.Series, pd.Series]:
    in_left = left.index.isin(left.dropna(how="all").index)
    in_right = right.index.isin(right.dropna(how="all").index)
    status = pd.Series("unchanged", index=left.index, dtype="string")
    status[(~in_left) & (in_right)] = "added"
    status[(in_left) & (~in_right)] = "removed"

    changed_cols = pd.Series(
        [[] for _ in range(len(left))], index=left.index, dtype=object
    )
    mask_overlap = in_left & in_right
    if compare_cols:
        for col in compare_cols:
            l = left[col] if col in left.columns else pd.Series(index=left.index, dtype=object)
            r = right[col] if col in right.columns else pd.Series(index=right.index, dtype=object)
            eq = pd.Series(True, index=left.index)
            eq.loc[mask_overlap] = _equal_values(
                l[mask_overlap], r[mask_overlap], precision
            )
            is_changed = (~eq) & mask_overlap
            if is_changed.any():
                status.loc[is_changed & (status == "unchanged")] = "changed"
                # append col name to changed_cols list
                idxs = is_changed[is_changed].index
                for i in idxs:
                    changed_cols.at[i] = [*changed_cols.at[i], col]
    return status, changed_cols


def _build_changes_table(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key_cols: Sequence[str],
    compare_cols: Sequence[str],
    options: CompareOptions,
) -> pd.DataFrame:
    idx = left.index
    status, changed_cols = _compute_status(
        left, right, compare_cols, options.number_precision
    )

    out = pd.DataFrame(index=idx)

    # key columns reconstructed from index; if an extra "__row__" helper level exists,
    # drop it from the display keys
    if isinstance(idx, pd.MultiIndex):
        key_frame = idx.to_frame(index=False)
        # Keep only the first len(key_cols) levels for display
        key_df = key_frame.iloc[:, : len(key_cols)].copy()
        key_df.columns = list(key_cols)
        # align index with output to avoid NaNs from index-mismatch during assignment
        key_df.index = idx
    else:
        key_df = pd.DataFrame({key_cols[0]: idx})

    out[list(key_cols)] = key_df
    out["__status__"] = status.values
    out["__changed_cols__"] = changed_cols.values

    # For each compare col add before/after and optional delta
    for col in compare_cols:
        l = left[col] if col in left.columns else pd.Series(index=idx, dtype=object)
        r = right[col] if col in right.columns else pd.Series(index=idx, dtype=object)
        out[f"{col} ({options.id_label})"] = l
        out[f"{col} ({options.new_label})"] = r
        if options.show_delta_cols:
            ln = _coerce_numeric(l)
            rn = _coerce_numeric(r)
            out[f"{col} Δ"] = rn - ln

    return out.reset_index(drop=True)


def _auto_col_widths(df: pd.DataFrame, max_width: int = 60) -> List[int]:
    widths = []
    for col in df.columns:
        series = df[col]
        # Sample first 500 rows for performance
        sample = series.astype(str).head(500)
        width = max([len(col) + 2, *(len(s) + 1 for s in sample)])
        widths.append(min(width, max_width))
    return widths


def _write_table_with_formats(
    writer: pd.ExcelWriter,
    df: pd.DataFrame,
    sheet_name: str,
    freeze_cols: int = 1,
    freeze_rows: int = 1,
    apply_filters: bool = True,
    table_style: str = "Table Style Light 9",
) -> None:
    df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1, header=False)
    wb = writer.book
    ws = writer.sheets[sheet_name]
    nrows, ncols = df.shape

    # Header format
    hdr = wb.add_format(
        {"bold": True, "text_wrap": False, "valign": "bottom", "border": 0}
    )
    for c, colname in enumerate(df.columns):
        ws.write(0, c, colname, hdr)

    # Create a table (gives filters and nice banding)
    if apply_filters and nrows > 0:
        ws.add_table(
            0,
            0,
            nrows,
            ncols - 1,
            {"style": table_style, "columns": [{"header": c} for c in df.columns]},
        )

    # Freeze panes
    ws.freeze_panes(freeze_rows, freeze_cols)

    # Column widths
    widths = _auto_col_widths(df)
    for c, w in enumerate(widths):
        ws.set_column(c, c, w)


def _apply_conditional_formats(
    writer: pd.ExcelWriter, df: pd.DataFrame, sheet_name: str, options: CompareOptions
) -> None:
    wb = writer.book
    ws = writer.sheets[sheet_name]
    nrows, ncols = df.shape

    # Status-based row shading
    fmt_added = wb.add_format({"bg_color": "#E8F5E9"})  # light green
    fmt_removed = wb.add_format({"bg_color": "#FFEBEE"})  # light red
    fmt_changed = wb.add_format({"bg_color": "#FFF8E1"})  # light amber

    try:
        status_col = df.columns.get_loc("__status__")
    except KeyError:
        return

    status_range = (1, status_col, nrows, status_col)  # data rows only

    ws.conditional_format(
        status_range[0],
        status_range[1],
        status_range[2],
        status_range[3],
        {
            "type": "text",
            "criteria": "containing",
            "value": "added",
            "format": fmt_added,
        },
    )
    ws.conditional_format(
        status_range[0],
        status_range[1],
        status_range[2],
        status_range[3],
        {
            "type": "text",
            "criteria": "containing",
            "value": "removed",
            "format": fmt_removed,
        },
    )
    ws.conditional_format(
        status_range[0],
        status_range[1],
        status_range[2],
        status_range[3],
        {
            "type": "text",
            "criteria": "containing",
            "value": "changed",
            "format": fmt_changed,
        },
    )

    # Highlight cell-level differences for each pair of before/after columns
    for i, col in enumerate(df.columns):
        if col.endswith(")") and (" (" in col):
            base = col.split(" (")[0]
            before_col = f"{base} ({options.id_label})"
            after_col = f"{base} ({options.new_label})"
            if before_col in df.columns and after_col in df.columns:
                c1 = df.columns.get_loc(before_col)
                c2 = df.columns.get_loc(after_col)
                rng1 = (1, c1, nrows, c1)
                rng2 = (1, c2, nrows, c2)
                # Yellow fill where values differ
                fmt_diff = wb.add_format({"bg_color": "#FFF59D"})
                ws.conditional_format(
                    rng1[0],
                    rng1[1],
                    rng1[2],
                    rng1[3],
                    {
                        "type": "formula",
                        "criteria": "=INDIRECT(ADDRESS(ROW(),%d))<>INDIRECT(ADDRESS(ROW(),%d))"
                        % (c2 + 1, c1 + 1),
                        "format": fmt_diff,
                    },
                )
                ws.conditional_format(
                    rng2[0],
                    rng2[1],
                    rng2[2],
                    rng2[3],
                    {
                        "type": "formula",
                        "criteria": "=INDIRECT(ADDRESS(ROW(),%d))<>INDIRECT(ADDRESS(ROW(),%d))"
                        % (c1 + 1, c2 + 1),
                        "format": fmt_diff,
                    },
                )

    # Delta numeric icon sets
    if options.delta_iconset:
        for col in df.columns:
            if col.endswith(" Δ"):
                c = df.columns.get_loc(col)
                ws.conditional_format(
                    1,
                    c,
                    nrows,
                    c,
                    {
                        "type": "icon_set",
                        "icon_style": "3_arrows",
                    },
                )


def _write_readme_and_summary(
    writer: pd.ExcelWriter,
    key_cols: Sequence[str],
    compare_cols: Sequence[str],
    options: CompareOptions,
    counts: Mapping[str, int],
    bycol_sheet_names: Mapping[str, str],
) -> None:
    wb = writer.book

    # README & Legend (simple text write)
    ws = wb.add_worksheet(options.readme_name)
    text = [
        ["Compare Workbook (flowplan)", ""],
        ["Baseline label:", options.id_label],
        ["Current label:", options.new_label],
        ["Key columns:", ", ".join(key_cols)],
        ["Compared columns:", ", ".join(compare_cols)],
        ["Legend:", ""],
        ["added", "Row exists only in Current"],
        ["removed", "Row exists only in Baseline"],
        ["changed", "Row exists in both but one or more compared columns differ"],
        ["unchanged", "Row exists in both and all compared columns equal"],
    ]
    bold = wb.add_format({"bold": True})
    normal = wb.add_format({})
    row = 0
    for label, val in text:
        ws.write(row, 0, label, bold)
        ws.write(row, 1, val, normal)
        row += 1

    # Summary sheet with counts & hyperlinks
    df_summary = pd.DataFrame(
        {
            "Metric": [
                "Total rows (union)",
                "Added rows",
                "Removed rows",
                "Changed rows",
            ],
            "Count": [
                counts.get("total", 0),
                counts.get("added", 0),
                counts.get("removed", 0),
                counts.get("changed", 0),
            ],
            "Go to": ["#", "#", "#", "#"],
        }
    )

    # include per-column links
    for col, sheet in bycol_sheet_names.items():
        df_summary.loc[len(df_summary)] = [
            f"Changed — {col}",
            None,
            f'=HYPERLINK("#{sheet}!A1", "Open")',
        ]

    df_summary["Go to"] = df_summary["Go to"].where(
        df_summary["Metric"] == "Total rows (union)", None
    )

    # hyperlink rows for Added/Removed/Changed
    links_map = {
        "Added rows": options.new_rows_name,
        "Removed rows": options.missing_rows_name,
        "Changed rows": options.changes_name,
    }
    for i, metric in enumerate(df_summary["Metric"]):
        sheet = links_map.get(metric)
        if sheet:
            df_summary.at[i, "Go to"] = f'=HYPERLINK("#{sheet}!A1", "Open")'

    _write_table_with_formats(
        writer, df_summary, options.summary_name, freeze_cols=0, freeze_rows=1
    )


def write_compare_workbook(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    key_cols: Sequence[str],
    compare_cols: Sequence[str],
    path: str,
    *,
    id_label: str = "Baseline",
    new_label: str = "Current",
    include_unchanged_in_bycol: bool = False,
    show_delta_cols: bool = True,
    number_precision: int = 6,
    delta_iconset: bool = True,
) -> None:
    """Create a readable Excel workbook showing differences between two DataFrames."""
    options = CompareOptions(
        id_label=id_label,
        new_label=new_label,
        include_unchanged_in_bycol=include_unchanged_in_bycol,
        show_delta_cols=show_delta_cols,
        number_precision=number_precision,
        delta_iconset=delta_iconset,
    )

    # Ensure path exists
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prep
    left, right, _ = _prep_dfs(df_left, df_right, key_cols)

    # Build master changes table
    master = _build_changes_table(left, right, key_cols, compare_cols, options)

    # Status metrics
    counts = {
        "total": len(master),
        "added": int((master["__status__"] == "added").sum()),
        "removed": int((master["__status__"] == "removed").sum()),
        "changed": int((master["__status__"] == "changed").sum()),
    }

    # Derived views
    df_changes = master[master["__status__"] == "changed"].copy()
    df_added = master[master["__status__"] == "added"].copy()
    df_removed = master[master["__status__"] == "removed"].copy()

    # Per-column narrow tables (before/after [+ delta])
    bycol_sheet_names = {}
    bycol_tables = {}
    for col in compare_cols:
        cols = [
            *key_cols,
            "__status__",
            f"{col} ({options.id_label})",
            f"{col} ({options.new_label})",
        ]
        if options.show_delta_cols:
            cols.append(f"{col} Δ")
        tbl = master[cols].copy()
        if not options.include_unchanged_in_bycol:
            tbl = tbl[tbl["__status__"] == "changed"]
        bycol_sheet_names[col] = f"Changed — {col}"[:31]  # Excel 31-char limit
        bycol_tables[col] = tbl

    with pd.ExcelWriter(
        str(output_path),
        engine="xlsxwriter",
        engine_kwargs={"options": {"strings_to_numbers": False}},
    ) as writer:
        # README & Summary
        _write_readme_and_summary(
            writer, key_cols, compare_cols, options, counts, bycol_sheet_names
        )

        # Changes sheets
        _write_table_with_formats(writer, df_changes, options.changes_name)
        _apply_conditional_formats(writer, df_changes, options.changes_name, options)

        _write_table_with_formats(writer, df_added, options.new_rows_name)
        _apply_conditional_formats(writer, df_added, options.new_rows_name, options)

        _write_table_with_formats(writer, df_removed, options.missing_rows_name)
        _apply_conditional_formats(writer, df_removed, options.missing_rows_name, options)

        # By-column sheets
        for col, tbl in bycol_tables.items():
            sheet = bycol_sheet_names[col]
            _write_table_with_formats(writer, tbl, sheet)
            _apply_conditional_formats(writer, tbl, sheet, options)