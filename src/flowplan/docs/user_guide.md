# FlowPlan: The Complete User Guide

**Version:** 1.0+
**Architecture:** Modular, Plugin-Ready, Optimized

---

## Table of Contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [The Configuration Ecosystem](#3-the-configuration-ecosystem)
4. [Step 1: Schemas & Sources](#4-step-1-schemas--sources)
5. [Step 2: Compilation Strategies](#5-step-2-compilation-strategies)
6. [Step 3: Enrichment & Fuzzy Matching](#6-step-3-enrichment--fuzzy-matching)
7. [Step 4: The Expression Engine (AST)](#7-step-4-the-expression-engine-ast)
8. [Step 5: Compare & Export](#8-step-5-compare--export)
9. [Advanced: Plugin System](#9-advanced-plugin-system)

---

## 1. Overview

**FlowPlan** is a declarative ETL (Extract, Transform, Load) framework designed to ingest messy data from Excel, Access, and JSON, normalize it into strictly typed DataFrames, and output clean, compiled reports.

### Key Capabilities

* **Zero-Code Logic**
  All joins, filters, and computed columns are defined declaratively in JSON.

* **Strict Typing**
  Enforces schema validation (e.g., nullable `Int64` vs `Float64`).

* **Vectorized Fuzzy Matching**
  Probabilistically joins datasets on messy text keys using a high-performance engine that operates only on unique values.

* **Audit Trails**
  Generates Excel workbooks highlighting exactly what changed between datasets (Added / Removed / Changed).

---

## 2. Installation

The library follows modern Python packaging standards using `pyproject.toml`.

### Standard Install

```bash
pip install flowplan
```

### With Access DB Support (Windows Only)

The core library is OS-agnostic. To enable Microsoft Access support (`.accdb` / `.mdb`):

```bash
pip install flowplan[access]
```

### For Developers (Editable Mode)

For extension development or running from source:

```bash
git clone <repo_url>
cd <repo_folder>
pip install -e .[dev,access]
```

---

## 3. The Configuration Ecosystem

The heart of FlowPlan is its configuration loader. Rather than relying on monolithic JSON files, the system supports a **modular configuration hierarchy**.

### Modular Configs & Includes

Definitions can be split by concern (schemas, sources, compilation, exports).

**File:** `config/main.json`

```json
{
  "version": "1.0",
  "include": [
    "./schema.json",
    "./sources/*.json",
    "./compile/hr_targets.json"
  ],
  "output": "./build"
}
```

### Variables & Profiles

Configurations support variable interpolation and profile-based overrides.

* Variable syntax: `${VAR_NAME}` or `${json.path}`

**Example:**

```json
"sources": [
  {
    "id": "hr_data",
    "path": "${DATA_ROOT}/hr_2024.xlsx"
  }
]
```

#### Running with Profiles

Profiles are JSON files located under `config/profiles/` and override keys in the base config.

```bash
# Loads config/main.json and merges config/profiles/prod.json
flowplan run --config config/main.json --profile prod
```

---

## 4. Step 1: Schemas & Sources

### The Schema (`schema.aliases`)

The schema acts as the **gold standard**. All input columns are coerced immediately upon load.

Supported constraints:

* `integer` — Pandas `Int64` (nullable)
* `string` — Pandas `string`
* `identifier: true` — Enforces uniqueness
* `enum` — Validates against an allowed set
* `not_null` — Raises an error if missing

**Example:**

```json
"schema": {
  "aliases": {
    "emp_id": { "type": "integer", "identifier": true },
    "status": { "type": "string", "enum": ["Active", "Inactive"] }
  }
}
```

### Source Adapters

Supported adapters:

* **Excel** — `.xlsx`, `.xls`, `.xlsb`
* **Access** — `.accdb`, `.mdb`
* **Inline** — Raw JSON arrays (ideal for lookup tables)

**Example Source Definition:**

```json
{
  "id": "raw_excel",
  "path": "data/input.xlsx",
  "tables": [
    {
      "name": "Sheet1",
      "table_id": "clean_sheet",
      "columns": {
        "Raw Name": {
          "alias": "clean_name",
          "transforms": [{ "titlecase": {} }]
        },
        "Emp ID": {
          "alias": "emp_id",
          "transforms": [{ "cast": { "to": "integer" } }]
        }
      }
    }
  ]
}
```

---

## 5. Step 2: Compilation Strategies

The `compile` section transforms raw sources into finalized **Targets**.

### Union (Stacking)

Stacks datasets vertically.

* `dedupe_on` — Columns used for deduplication
* `keep` — `first` or `last`
* `add_source` — Adds a `_source` column

### Merge (Joining)

Joins datasets horizontally on a primary key.

* `merge_rules` — Resolves column collisions
* `strategy: first_non_null` — First value across inputs
* `strategy: prefer_source` — Prioritizes a named input

### Diff (Delta Analysis)

Compares two datasets to identify changes.

* `side`: `left_only`, `right_only`, `both`, `symmetric_diff`

**Diff Example:**

```json
{
  "name": "new_employees",
  "mode": "diff",
  "left": "roster_jan",
  "right": "roster_feb",
  "on": ["employee_id"],
  "side": "right_only"
}
```

---

## 6. Step 3: Enrichment & Fuzzy Matching

Enrichment allows approximate joins against reference datasets using fuzzy matching.

### Performance Optimization

The optimized vectorized matcher:

1. Extracts unique values from the left-side key
2. Runs expensive fuzzy logic only on unique values
3. Broadcasts results back to original rows

**Impact:** 100,000 rows with 50 unique keys can run **100×–2000× faster** than naive approaches.

### Configuration Example

```json
"enrich": [
  {
    "from": "master_companies",
    "left_on": "raw_company_name",
    "right_on": "canonical_name",
    "add": { "company_id": "id" },
    "match": {
      "strategy": ["exact", "normalized", "fuzzy"],
      "normalize": ["strip", "lower", "collapse_ws", "strip_punct"],
      "fuzzy": {
        "scorer": "token_set_ratio",
        "threshold": 90,
        "block": "first_char"
      },
      "audit": true
    }
  }
]
```

---

## 7. Step 4: The Expression Engine (AST)

Filters and computed columns are evaluated using a safe **Abstract Syntax Tree (AST)**. No `eval()` is used.

### Syntax

* Array form: `["operator", arg1, arg2, ...]`
* Object form: `{ "op": "operator", "args": [...] }`

### Available Operators

| Category | Operators                                |
| -------- | ---------------------------------------- |
| Math     | add, sub, mul, div, round, clip, percent |
| Logic    | and, or, not, if, eq, gt, lt, neq        |
| String   | concat, len, regex_replace               |
| Date     | datediff (day, hour), strftime           |
| Nulls    | coalesce, fillna, is_null                |

**Example:**

```json
"Total Comp": {
  "compute": [
    "if",
    ["eq", {"col": "status"}, "Active"],
    ["add", {"col": "salary"}, {"col": "bonus"}],
    0
  ]
}
```

---

## 8. Step 5: Compare & Export

### Compare Reports

Generates Excel workbooks with color-coded tabs:

* **Summary** — Counts of changes
* **Changes** — Side-by-side diffs
* **New / Missing** — Added or removed rows

```json
"compare": {
  "pairs": [
    {
      "left": "snapshot_yesterday",
      "right": "snapshot_today",
      "on": ["id"],
      "compare_cols": ["status", "amount"],
      "save_name": "Daily_Change_Report"
    }
  ]
}
```

### Exporting

Produces the final artifacts:

* `sheets` — Multiple tabs
* `columns` — Reorder, rename, compute
* `filter` — AST-based row filtering

---

## 9. Advanced: Plugin System

FlowPlan is designed to be extended without forking.

Extensions are registered using Python `entry_points` in `pyproject.toml`.

### Extending via `pyproject.toml`

```toml
[project.entry-points."flowplan.transforms"]
reverse_string = "my_company_pkg.utils:reverse_string"

[project.entry-points."flowplan.expr_ops"]
business_days = "my_company_pkg.dates:calc_business_days"

[project.entry-points."flowplan.enrich"]
ml_predict_category = "my_company_pkg.ml:predict"
```

### UDF Signature Requirements

**Transform Function**

```python
def my_transform(series: pd.Series, params: dict) -> pd.Series:
    return series.apply(...)
```

**Expression Operator**

```python
def my_op(df: pd.DataFrame, *args) -> pd.Series:
    return args[0] + args[1]
```
