#!/usr/bin/env python3
"""Classify deed grantee legal form using the published ownership model.

The published project retained production predictions for 160,000+ normalized
Douglas County owner names, but the serialized BERT weight file is no longer
present locally. Exact name matches therefore use the original stored BERT
prediction and confidence. A deterministic character model distilled from the
stored predictions classifies previously unseen deed grantee strings. The
published labeled corpus is added to the final fit and is also used for a
separate reference-label audit. No new human-coding round is required.

Row-level names and predictions remain under ignored ``data_work/`` paths.
Tracked diagnostics contain only counts, metrics, hashes, and class labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PUBLISHED_ROOT = PROJECT_ROOT / "data/external/ownership_model"
DEFAULT_TRAINING = (
    PUBLISHED_ROOT / "data/processed/standardized_training_dataset_final.csv"
)
DEFAULT_PUBLISHED_METRICS = (
    PUBLISHED_ROOT
    / "results/classification_results/cv_run/bert_cv_oof_summary.json"
)
DEFAULT_PUBLISHED_PREDICTIONS = (
    PUBLISHED_ROOT
    / "results/production_classification_results/owner_type_mapping_20250802_192450.csv"
)
PUBLISHED_MODEL_DIR = PUBLISHED_ROOT / "models/advanced_optimized_model_fold_1"
PUBLISHED_WEIGHT_PATH = PUBLISHED_MODEL_DIR / "model.safetensors"

SALES_PATH = PROJECT_ROOT / "data_work/transactions_linked.parquet"
DISTANCE_PATH = PROJECT_ROOT / "data_work/parcel_boundary_distances.parquet"
RESTRICTED_DIR = PROJECT_ROOT / "data_work/restricted/grantee_classification"
PREDICTION_PATH = RESTRICTED_DIR / "grantee_model_predictions.parquet"
LABEL_PATH = RESTRICTED_DIR / "grantee_model_labels.parquet"
AUDIT_PATH = PROJECT_ROOT / "data_work/diagnostics/grantee_classifier_transfer_audit.csv"
SUMMARY_PATH = PROJECT_ROOT / "data_work/diagnostics/grantee_classifier_summary.csv"

LABELS = ("Individual", "LLC", "Corporation", "Trust", "Other")
EVENT_DATE = pd.Timestamp("2019-03-13")
WINDOW_START = pd.Timestamp("2015-03-13")
WINDOW_END_EXCLUSIVE = pd.Timestamp("2022-03-13")
RANDOM_SEED = 42


def ensure_env() -> None:
    if not os.getenv("VIRTUAL_ENV") or not os.getenv("VIRTUAL_ENV", "").endswith("/.venv"):
        raise SystemExit("Activate the project .venv before running")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_name(values: pd.Series) -> pd.Series:
    """Normalize owner strings for deterministic exact matching and hashing."""
    return (
        values.astype("string")
        .fillna("")
        .str.upper()
        .str.replace("&", " AND ", regex=False)
        .str.replace(r"[^0-9A-Z]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def grantee_key(normalized_name: str) -> str:
    return hashlib.sha256(normalized_name.encode("utf-8")).hexdigest()


def _clean_codes(values: pd.Series) -> pd.Series:
    """Normalize class labels; retained as a public validation utility."""
    aliases = {label.upper(): label for label in LABELS}
    normalized = values.astype("string").fillna("").str.strip().str.upper()
    return normalized.map(aliases).fillna("")


def build_reference_model() -> Pipeline:
    """Small independently evaluated transfer model for the labeled corpus."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 6),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    C=4.0,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def build_distilled_model() -> Pipeline:
    """Reproduce the published BERT classifications without its missing weights."""
    return Pipeline(
        [
            (
                "hashing",
                HashingVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 6),
                    n_features=2**20,
                    alternate_sign=False,
                    norm="l2",
                    lowercase=True,
                ),
            ),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    alpha=3e-7,
                    max_iter=100,
                    tol=1e-4,
                    average=True,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def load_training(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"owner_name", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Training data missing fields: {sorted(missing)}")
    frame = frame.loc[frame["label"].isin(LABELS), ["owner_name", "label"]].copy()
    frame["owner_name"] = frame["owner_name"].astype("string").str.strip()
    frame["name_group"] = normalize_name(frame["owner_name"])
    if frame["owner_name"].eq("").any() or frame["name_group"].eq("").any():
        raise ValueError("Training data contain blank owner names")
    conflicted = frame.groupby("name_group")["label"].nunique()
    if conflicted.gt(1).any():
        raise ValueError(
            f"Training data contain {int(conflicted.gt(1).sum())} normalized-name label conflicts"
        )
    return frame


def load_published_predictions(path: Path) -> tuple[pd.DataFrame, int]:
    frame = pd.read_csv(
        path,
        usecols=["Current_Ow", "predicted_owner_type", "prediction_confidence"],
    ).rename(
        columns={
            "Current_Ow": "owner_name",
            "predicted_owner_type": "published_class",
            "prediction_confidence": "published_confidence",
        }
    )
    frame["published_class"] = _clean_codes(frame["published_class"])
    frame["published_confidence"] = pd.to_numeric(
        frame["published_confidence"], errors="coerce"
    )
    frame["name_group"] = normalize_name(frame["owner_name"])
    frame = frame.loc[
        frame["name_group"].ne("")
        & frame["published_class"].isin(LABELS)
        & frame["published_confidence"].between(0, 1, inclusive="both")
    ].copy()
    conflicts = frame.groupby("name_group")["published_class"].nunique()
    conflicted_groups = set(conflicts.loc[conflicts.gt(1)].index)
    frame = frame.loc[~frame["name_group"].isin(conflicted_groups)].copy()
    frame = (
        frame.sort_values(
            ["name_group", "published_confidence"], ascending=[True, False]
        )
        .drop_duplicates("name_group")
        .reset_index(drop=True)
    )
    return frame, len(conflicted_groups)


def _classification_rows(
    section: str,
    truth: pd.Series | np.ndarray,
    prediction: pd.Series | np.ndarray,
) -> list[dict[str, object]]:
    truth = pd.Series(truth, dtype="string")
    prediction = pd.Series(prediction, dtype="string")
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=list(LABELS), zero_division=0
    )
    rows: list[dict[str, object]] = [
        {"section": section, "metric": "rows", "label": "all", "value": len(truth)},
        {
            "section": section,
            "metric": "accuracy",
            "label": "all",
            "value": accuracy_score(truth, prediction),
        },
        {
            "section": section,
            "metric": "macro_f1",
            "label": "all",
            "value": f1_score(truth, prediction, average="macro"),
        },
        {
            "section": section,
            "metric": "weighted_f1",
            "label": "all",
            "value": f1_score(truth, prediction, average="weighted"),
        },
    ]
    for index, label in enumerate(LABELS):
        rows.extend(
            [
                {"section": section, "metric": "precision", "label": label, "value": precision[index]},
                {"section": section, "metric": "recall", "label": label, "value": recall[index]},
                {"section": section, "metric": "f1", "label": label, "value": f1[index]},
                {"section": section, "metric": "support", "label": label, "value": support[index]},
            ]
        )
    return rows


def evaluate_reference_model(training: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    model = build_reference_model()
    predictions = cross_val_predict(
        model,
        training["owner_name"],
        training["label"],
        groups=training["name_group"],
        cv=splitter,
        method="predict",
        n_jobs=1,
    )
    audit = pd.DataFrame(
        _classification_rows("reference_model_grouped_oof", training["label"], predictions)
    )
    model.fit(training["owner_name"], training["label"])
    return model, audit


def fit_and_audit_distilled_model(
    training: pd.DataFrame,
    published: pd.DataFrame,
) -> tuple[Pipeline, pd.DataFrame]:
    """Audit on withheld teacher names and labeled names, then fit all evidence."""
    human_groups = set(training["name_group"])
    external = published.loc[~published["name_group"].isin(human_groups)].copy()
    train_index, test_index = train_test_split(
        np.arange(len(external)),
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=external["published_class"],
    )
    audit_model = build_distilled_model()
    audit_model.fit(
        external.iloc[train_index]["owner_name"],
        external.iloc[train_index]["published_class"],
    )
    teacher_prediction = audit_model.predict(external.iloc[test_index]["owner_name"])
    human_prediction = audit_model.predict(training["owner_name"])
    rows = _classification_rows(
        "distilled_model_teacher_holdout",
        external.iloc[test_index]["published_class"],
        teacher_prediction,
    )
    rows.extend(
        _classification_rows(
            "distilled_model_labeled_reference",
            training["label"],
            human_prediction,
        )
    )

    final_text = pd.concat(
        [external["owner_name"], training["owner_name"]], ignore_index=True
    )
    final_label = pd.concat(
        [external["published_class"], training["label"]], ignore_index=True
    )
    final_model = build_distilled_model()
    final_model.fit(final_text, final_label)
    rows.append(
        {
            "section": "distilled_model_final_fit",
            "metric": "rows",
            "label": "all",
            "value": len(final_text),
        }
    )
    return final_model, pd.DataFrame(rows)


def target_grantees() -> tuple[pd.DataFrame, pd.DataFrame]:
    sales = pd.read_parquet(
        SALES_PATH,
        columns=["sale_date", "parcel_id", "grantee", "input_row_valid"],
    )
    distance = pd.read_parquet(
        DISTANCE_PATH,
        columns=[
            "parcel_id",
            "signed_dist_sfha_m",
            "signed_dist_inund_m",
            "inside_sfha",
            "inside_inund",
        ],
    )
    sales["parcel_id"] = sales["parcel_id"].astype("string")
    distance["parcel_id"] = distance["parcel_id"].astype("string")
    sales["sale_date"] = pd.to_datetime(sales["sale_date"], errors="coerce")
    frame = sales.merge(distance, on="parcel_id", how="inner", validate="many_to_one")
    frame = frame.loc[
        frame["input_row_valid"].fillna(False)
        & frame["sale_date"].ge(WINDOW_START)
        & frame["sale_date"].lt(WINDOW_END_EXCLUSIVE)
        & frame["grantee"].notna()
    ].copy()
    frame["grantee_name"] = frame["grantee"].astype("string").str.strip()
    frame = frame.loc[frame["grantee_name"].ne("")].copy()
    frame["in_sfha_window"] = frame["signed_dist_sfha_m"].abs().le(300)
    frame["in_inund_window"] = frame["signed_dist_inund_m"].abs().le(300)
    frame = frame.loc[frame["in_sfha_window"] | frame["in_inund_window"]].copy()
    frame["name_group"] = normalize_name(frame["grantee_name"])
    frame["grantee_key"] = frame["name_group"].map(grantee_key)
    frame["post"] = frame["sale_date"].ge(EVENT_DATE)

    summary = (
        frame.groupby(["grantee_key", "name_group"], as_index=False)
        .agg(
            grantee_name=("grantee_name", "first"),
            sale_records=("grantee_key", "size"),
            sfha_window_records=("in_sfha_window", "sum"),
            inund_window_records=("in_inund_window", "sum"),
            inside_sfha_records=("inside_sfha", "sum"),
            inside_inund_records=("inside_inund", "sum"),
            pre_records=("post", lambda values: int((~values).sum())),
            post_records=("post", "sum"),
        )
        .sort_values("grantee_key")
        .reset_index(drop=True)
    )
    if summary["grantee_key"].duplicated().any():
        raise AssertionError("Target file does not have one row per normalized grantee")
    return frame, summary


def classify_targets(
    summary: pd.DataFrame,
    published: pd.DataFrame,
    model: Pipeline,
) -> pd.DataFrame:
    probabilities = model.predict_proba(summary["grantee_name"])
    classes = model.named_steps["classifier"].classes_
    surrogate_index = probabilities.argmax(axis=1)
    result = summary.copy()
    result["surrogate_class"] = classes[surrogate_index]
    result["surrogate_score"] = probabilities.max(axis=1)
    for index, label in enumerate(classes):
        result[f"surrogate_probability_{str(label).lower()}"] = probabilities[:, index]

    result = result.merge(
        published[["name_group", "published_class", "published_confidence"]],
        on="name_group",
        how="left",
        validate="one_to_one",
    )
    exact = result["published_class"].notna()
    result["final_class"] = result["surrogate_class"]
    result.loc[exact, "final_class"] = result.loc[exact, "published_class"]
    result["model_confidence"] = result["surrogate_score"]
    result.loc[exact, "model_confidence"] = result.loc[exact, "published_confidence"]
    result["classification_source"] = np.where(
        exact,
        "published_bert_production_cache_exact_name",
        "distilled_published_classifier_unseen_name",
    )
    result["score_type"] = np.where(
        exact,
        "published_bert_softmax",
        "distilled_logistic_score_not_calibrated",
    )
    return result


def generate_classifications(
    training_path: Path,
    published_metrics_path: Path | None,
    published_predictions_path: Path,
) -> None:
    training = load_training(training_path)
    published, conflict_count = load_published_predictions(published_predictions_path)
    _, reference_audit = evaluate_reference_model(training)
    model, distilled_audit = fit_and_audit_distilled_model(training, published)
    target_rows, summary = target_grantees()
    predictions = classify_targets(summary, published, model)

    if predictions["grantee_key"].duplicated().any():
        raise AssertionError("Model output contains duplicate grantee keys")
    if not predictions["final_class"].isin(LABELS).all():
        raise AssertionError("Model output contains an invalid class")
    if predictions["model_confidence"].isna().any():
        raise AssertionError("Model output contains missing scores")
    if len(predictions) != len(summary):
        raise AssertionError("Model output does not cover every target grantee")

    RESTRICTED_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(PREDICTION_PATH, index=False)
    predictions[
        ["grantee_key", "final_class", "model_confidence", "classification_source", "score_type"]
    ].to_parquet(LABEL_PATH, index=False)

    exact = predictions["classification_source"].eq(
        "published_bert_production_cache_exact_name"
    )
    audit = pd.concat([reference_audit, distilled_audit], ignore_index=True)
    source_rows: list[dict[str, object]] = [
        {"section": "source", "metric": "training_sha256", "label": "all", "value": sha256_file(training_path)},
        {"section": "source", "metric": "published_predictions_sha256", "label": "all", "value": sha256_file(published_predictions_path)},
        {"section": "source", "metric": "published_prediction_name_groups", "label": "all", "value": len(published)},
        {"section": "source", "metric": "discarded_conflicting_name_groups", "label": "all", "value": conflict_count},
        {"section": "source", "metric": "published_weight_artifact_present", "label": "all", "value": PUBLISHED_WEIGHT_PATH.exists()},
        {"section": "target", "metric": "unique_grantees", "label": "all", "value": len(predictions)},
        {"section": "target", "metric": "deed_rows", "label": "all", "value": len(target_rows)},
        {"section": "target", "metric": "exact_published_name_matches", "label": "all", "value": int(exact.sum())},
        {"section": "target", "metric": "exact_published_deed_rows", "label": "all", "value": int(predictions.loc[exact, "sale_records"].sum())},
        {"section": "target", "metric": "distilled_name_predictions", "label": "all", "value": int((~exact).sum())},
        {"section": "target", "metric": "label_output_sha256", "label": "all", "value": sha256_file(LABEL_PATH)},
        {"section": "target", "metric": "promotion_status", "label": "all", "value": "ready_model_generated_labels"},
    ]
    audit = pd.concat([audit, pd.DataFrame(source_rows)], ignore_index=True)
    if published_metrics_path and published_metrics_path.exists():
        published_metrics = json.loads(published_metrics_path.read_text())
        for metric, value in published_metrics.get("test_holdout", {}).get("overall", {}).items():
            audit.loc[len(audit)] = {
                "section": "published_bert_holdout",
                "metric": metric,
                "label": "all",
                "value": value,
            }
        audit.loc[len(audit)] = {
            "section": "published_bert_holdout",
            "metric": "metrics_file_sha256",
            "label": "all",
            "value": sha256_file(published_metrics_path),
        }
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(AUDIT_PATH, index=False)

    summary_rows: list[dict[str, object]] = [
        {"metric": "unique_grantees", "label": "all", "value": len(predictions)},
        {"metric": "deed_rows_represented", "label": "all", "value": int(predictions["sale_records"].sum())},
        {"metric": "exact_published_name_matches", "label": "all", "value": int(exact.sum())},
        {"metric": "exact_published_deed_rows", "label": "all", "value": int(predictions.loc[exact, "sale_records"].sum())},
        {"metric": "model_score_at_least_0_80_names", "label": "all", "value": int(predictions["model_confidence"].ge(0.80).sum())},
        {"metric": "model_score_at_least_0_80_deeds", "label": "all", "value": int(predictions.loc[predictions["model_confidence"].ge(0.80), "sale_records"].sum())},
        {"metric": "classification_complete", "label": "all", "value": True},
    ]
    for label in LABELS:
        mask = predictions["final_class"].eq(label)
        summary_rows.extend(
            [
                {"metric": "unique_grantees_by_class", "label": label, "value": int(mask.sum())},
                {"metric": "deed_rows_by_class", "label": label, "value": int(predictions.loc[mask, "sale_records"].sum())},
            ]
        )
    pd.DataFrame(summary_rows).to_csv(SUMMARY_PATH, index=False)

    print(f"PASS: classified {len(predictions):,} unique grantees covering {len(target_rows):,} deed rows")
    print(
        "Exact stored BERT matches: "
        f"{int(exact.sum()):,} names / {int(predictions.loc[exact, 'sale_records'].sum()):,} deeds"
    )
    print(f"Wrote {LABEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {AUDIT_PATH.relative_to(PROJECT_ROOT)}")


def run(
    training_path: str | Path = DEFAULT_TRAINING,
    published_metrics_path: str | Path | None = DEFAULT_PUBLISHED_METRICS,
    published_predictions_path: str | Path = DEFAULT_PUBLISHED_PREDICTIONS,
) -> None:
    ensure_env()
    training = Path(training_path).expanduser().resolve()
    predictions = Path(published_predictions_path).expanduser().resolve()
    metrics = (
        Path(published_metrics_path).expanduser().resolve()
        if published_metrics_path
        else None
    )
    if not training.exists():
        raise SystemExit(f"Published-project training corpus not found: {training}")
    if not predictions.exists():
        raise SystemExit(f"Published-project production predictions not found: {predictions}")
    generate_classifications(training, metrics, predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", default=str(DEFAULT_TRAINING))
    parser.add_argument("--published-metrics", default=str(DEFAULT_PUBLISHED_METRICS))
    parser.add_argument(
        "--published-predictions", default=str(DEFAULT_PUBLISHED_PREDICTIONS)
    )
    args = parser.parse_args()
    run(args.training, args.published_metrics, args.published_predictions)


if __name__ == "__main__":
    main()
