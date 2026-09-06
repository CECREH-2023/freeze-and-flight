#!/usr/bin/env python3
"""
Transaction-mode provenance and validation helpers.

Counterpart to ``snapshot_mode.py``. Where snapshot mode documents an assessor
*latest-sale snapshot* (one retained sale per parcel), transaction mode documents
a complete *deed transaction ledger*: one row per delivered recorded sale, with
grantor (seller), grantee (buyer), recording reference, sale date, and price,
keyed to a parcel/APN. Source usability codes remain available so downstream
analysis can apply an explicit arm's-length rule.

A true ledger is REQUIRED before the project may make transaction-flow claims
(sale rate, liquidity, repeat-sales price, deed-level buyer composition). These
helpers implement a hard validation gate: a file that still has exactly one row
per parcel is flagged as a likely snapshot and must NOT be treated as a ledger.

Canonical internal sales schema (what ingest should produce)
------------------------------------------------------------
parcel_id        str      APN linking to assessor parcels (required)
sale_date        date     recording / sale date (required)
price_nominal    float    consideration / sale price in nominal dollars (required)
instrument       str      deed/instrument type (WD, QC, SW, ...) (recommended)
grantor          str      seller name(s) (recommended)
grantee          str      buyer name(s) (recommended; enables true buyer comp)
arms_length      bool     arms-length / qualified-sale flag if provided (optional)
source_record_id str      unique id from the source (Form 521 no., TransId) (optional)
sale_id          str      unique transaction id constructed at ingest (required)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

DATA_MODE = "deed_transaction_ledger"
SALE_RECORD_SCOPE = "all_delivered_residential_sale_records"
SALE_OUTCOME_DEFINITION = (
    "After an explicit analysis filter, sold_this_month equals one when a delivered "
    "recorded sale for the parcel falls in that month; transaction mode supports a "
    "genuine monthly transaction indicator."
)
BUYER_RECORD_SCOPE = "deed_grantee_at_transaction"

# Fields that must be present (post-mapping) for a file to qualify as a ledger.
REQUIRED_FIELDS = ["parcel_id", "sale_date", "price_nominal"]
# Fields that unlock the transaction-flow analyses; warn if absent.
RECOMMENDED_FIELDS = ["grantee", "grantor", "instrument"]


def _remove_appledouble(path: Path) -> None:
    """Remove the macOS metadata sidecar created beside a generated output."""
    sidecar = path.with_name(f"._{path.name}")
    sidecar.unlink(missing_ok=True)


def annotate_sales_transaction(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Attach transaction-mode provenance columns to a sales-like DataFrame."""
    out = df.copy()
    out["data_mode"] = DATA_MODE
    out["sale_record_scope"] = SALE_RECORD_SCOPE
    out["is_complete_transaction_ledger"] = True
    has_grantee = (
        "grantee" in df.columns
        and df["grantee"].fillna("").astype(str).str.strip().ne("").any()
    )
    out["buyer_fields_are_true_deed_grantee"] = has_grantee
    out["transaction_validation_stage"] = stage
    return out


def detect_data_mode(df: pd.DataFrame, id_col: str = "parcel_id") -> str:
    """Heuristically classify a sales file as 'transaction' or 'snapshot'.

    A file is 'transaction' only if at least one parcel has more than one sale
    row. Otherwise it is treated as a 'snapshot' regardless of column labels.
    """
    if id_col not in df.columns or df.empty:
        return "unknown"
    max_rows = int(df.groupby(id_col).size().max())
    return "transaction" if max_rows > 1 else "snapshot"


def transaction_sales_summary(
    df: pd.DataFrame,
    stage: str,
    id_col: str = "parcel_id",
    date_col: str = "sale_date",
) -> pd.DataFrame:
    """Return a one-row validation summary for a candidate transaction ledger."""
    counts = df.groupby(id_col).size() if id_col in df.columns and len(df) else pd.Series(dtype=int)
    max_rows = int(counts.max()) if len(counts) else 0
    parcels_multi = int((counts > 1).sum()) if len(counts) else 0
    missing_required = [c for c in REQUIRED_FIELDS if c not in df.columns]
    def has_values(column: str) -> bool:
        return (
            column in df.columns
            and df[column].fillna("").astype(str).str.strip().ne("").any()
        )

    missing_recommended = [c for c in RECOMMENDED_FIELDS if not has_values(c)]

    if missing_required:
        status = "fail_missing_required_fields"
    elif max_rows <= 1:
        # The hard gate: looks like an assessor snapshot, not a ledger.
        status = "warn_possible_snapshot_one_row_per_parcel"
    else:
        status = "pass_transaction_mode"

    dates = pd.to_datetime(df[date_col], errors="coerce") if date_col in df.columns else pd.Series(dtype="datetime64[ns]")
    return pd.DataFrame(
        [
            {
                "stage": stage,
                "data_mode": DATA_MODE,
                "sale_record_scope": SALE_RECORD_SCOPE,
                "rows": int(len(df)),
                "unique_parcels": int(df[id_col].nunique()) if id_col in df.columns else 0,
                "max_rows_per_parcel": max_rows,
                "median_rows_per_parcel": float(counts.median()) if len(counts) else 0.0,
                "parcels_with_multiple_sales": parcels_multi,
                "has_grantee": has_values("grantee"),
                "has_grantor": has_values("grantor"),
                "has_instrument": has_values("instrument"),
                "missing_required_fields": ", ".join(missing_required) or "none",
                "missing_recommended_fields": ", ".join(missing_recommended) or "none",
                "sale_date_min": dates.min().date().isoformat() if dates.notna().any() else pd.NA,
                "sale_date_max": dates.max().date().isoformat() if dates.notna().any() else pd.NA,
                "validation_status": status,
                "outcome_definition": SALE_OUTCOME_DEFINITION,
            }
        ]
    )


def assert_is_ledger(df: pd.DataFrame, stage: str = "sales", id_col: str = "parcel_id") -> pd.DataFrame:
    """Raise if ``df`` does not qualify as a transaction ledger; else return the summary.

    Implements the audit's hard gate: fail when the file has exactly one row per
    parcel (a snapshot) or is missing required fields.
    """
    summary = transaction_sales_summary(df, stage=stage, id_col=id_col)
    status = summary.iloc[0]["validation_status"]
    if status != "pass_transaction_mode":
        raise SystemExit(
            f"Transaction-mode validation FAILED ({status}). "
            f"max_rows_per_parcel={summary.iloc[0]['max_rows_per_parcel']}, "
            f"missing_required={summary.iloc[0]['missing_required_fields']}. "
            "Provide a complete deed ledger (one row per sale) before running "
            "transaction-mode analyses, or stay in snapshot mode."
        )
    return summary


def write_transaction_diagnostics(
    df: pd.DataFrame,
    stage: str,
    out_dir: Path = Path("data_work/diagnostics"),
    prefix: str = "transaction_mode_sales",
) -> pd.DataFrame:
    """Write CSV + Markdown validation diagnostics for a candidate ledger."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = transaction_sales_summary(df, stage=stage)
    csv_path = out_dir / f"{prefix}_validation.csv"
    summary.to_csv(csv_path, index=False)
    row = summary.iloc[0]
    md = [
        "# Transaction-Mode Sales Validation",
        "",
        f"Stage: `{stage}`",
        f"Data mode: `{DATA_MODE}`",
        f"Rows (sales): {row['rows']:,}",
        f"Unique parcels: {row['unique_parcels']:,}",
        f"Max sales per parcel: {row['max_rows_per_parcel']:,}",
        f"Parcels with multiple sales: {row['parcels_with_multiple_sales']:,}",
        f"Has grantee (true buyer): {row['has_grantee']}",
        f"Sale date range: {row['sale_date_min']} to {row['sale_date_max']}",
        f"Validation status: **{row['validation_status']}**",
        "",
        "A `pass_transaction_mode` status means repeat-sales price indices, true "
        "sale-rate/liquidity outcomes, and deed-grantee buyer composition become "
        "estimable. A `warn_possible_snapshot...` status means the file is still a "
        "latest-sale snapshot and transaction-flow claims remain unsupported.",
    ]
    md_path = out_dir / f"{prefix}_validation.md"
    md_path.write_text("\n".join(md) + "\n")
    _remove_appledouble(csv_path)
    _remove_appledouble(md_path)
    return summary


def write_no_data_status(path: Path, reason: str, status: str = "no_data", extra: Optional[dict] = None) -> None:
    row = {"status": status, "reason": reason}
    if extra:
        row.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)
