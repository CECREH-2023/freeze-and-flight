#!/usr/bin/env python3
"""Estimate descriptive changes in deed composition and prices after the 2019 Nebraska flood.

See docs/METHODS.md for the study population and interpretation limits.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import linalg, stats
from statsmodels.stats.multitest import multipletests
from wildboottest.weights import draw_weights

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260904
OUTCOMES = ("zero_improvement_share", "pooled_log_price", "improved_log_price")
CATEGORIES = ("positive", "zero", "unknown")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def event_year(dates, event="2019-03-13"):
    dates = pd.to_datetime(dates, errors="coerce")
    t = pd.Timestamp(event)
    before = (dates.dt.month < t.month) | ((dates.dt.month == t.month) & (dates.dt.day < t.day))
    return (dates.dt.year - t.year - before.astype(int)).astype("Int64")


def code(values):
    return pd.to_numeric(values, errors="coerce").astype("Int64")


def improvement_category(values):
    x = pd.to_numeric(values, errors="coerce")
    return pd.Series(np.select([np.isfinite(x) & x.gt(0), np.isfinite(x) & x.eq(0)], ["positive", "zero"], default="unknown"), index=values.index)


def bundle_flags(data):
    """Flag shared instruments before residential and geography restrictions."""
    d = data.copy()
    book = d.deed_book.astype("string").fillna("").str.strip()
    page = d.deed_page.astype("string").fillna("").str.strip()
    valid = book.ne("") & page.ne("") & ~book.isin(["0", "nan"]) & ~page.isin(["0", "nan"])
    key = book + "|" + page
    d["instrument_key"] = key.where(valid, "row|" + d.sale_id.astype(str))
    d["audit_parcel"] = d.parcel_id.astype("string")
    if "source_location_id" in d:
        d["audit_parcel"] = d.audit_parcel.fillna("source|" + d.source_location_id.astype("string"))
    d["audit_parcel"] = d.audit_parcel.fillna("unknown|" + d.sale_id.astype(str))
    d["instrument_parcels"] = d.groupby("instrument_key").audit_parcel.transform("nunique")
    d["instrument_dates"] = d.groupby("instrument_key").sale_date.transform("nunique")
    d["instrument_parcel_prices"] = d.groupby(["instrument_key", "audit_parcel"]).price_nominal.transform("nunique")
    # Equal amounts alone are common. The combination of the same date, both
    # parties, amount, and multiple parcels is an allocation-review candidate.
    party_cols = [x for x in ("grantor", "grantee") if x in d]
    d["repeated_party_amount_parcels"] = 1
    if len(party_cols) == 2:
        normalized = [d[x].astype("string").fillna("").str.upper().str.replace(r"\s+", " ", regex=True).str.strip() for x in party_cols]
        party_key = d.sale_date.dt.strftime("%Y-%m-%d").fillna("") + "|" + normalized[0] + "|" + normalized[1] + "|" + d.price_nominal.astype(str)
        party_ok = normalized[0].ne("") & normalized[1].ne("") & d.price_nominal.gt(0)
        d["party_amount_key"] = party_key.where(party_ok, "row|" + d.sale_id.astype(str))
        d["repeated_party_amount_parcels"] = d.groupby("party_amount_key").audit_parcel.transform("nunique")
    d["ambiguous_bundle"] = d.instrument_parcels.gt(1) | d.instrument_dates.gt(1) | d.instrument_parcel_prices.gt(1) | d.repeated_party_amount_parcels.gt(1)
    d["duplicate_instrument_parcel"] = d.duplicated(["instrument_key", "audit_parcel", "sale_date", "price_nominal"], keep="first")
    return d


def support(data):
    d = data.dropna(subset=["inside", "post", "cluster_1km"])
    cells = d.groupby(["inside", "post"]).size().reindex(pd.MultiIndex.from_product([[0, 1], [0, 1]]), fill_value=0)
    four = int(d[["cluster_1km", "inside", "post"]].drop_duplicates().groupby("cluster_1km").size().eq(4).sum())
    return {"n_obs": len(d), "min_cell": int(cells.min()), "four_cell_clusters": four,
            "support_gate": bool(cells.min() >= 20 and four >= 10),
            **{f"cell_{a}_{b}": int(cells.loc[(a, b)]) for a in (0, 1) for b in (0, 1)}}


def orthogonal_basis(controls):
    c = np.asarray(controls, float)
    if c.shape[1] == 0:
        return np.empty((len(c), 0))
    q, r, _ = linalg.qr(c, mode="economic", pivoting=True, check_finite=False)
    diagonal = np.abs(np.diag(r))
    tol = max(c.shape) * np.finfo(float).eps * (diagonal.max() if len(diagonal) else 0)
    rank = int((diagonal > tol).sum())
    return q[:, :rank]


def full_model_fit(y, x, controls, clusters, boot=9999, seed=SEED):
    """Exact FWL coefficient with full-projection CRV1 and WCR11 studentization.

    The bootstrap projects each restricted pseudo-outcome onto the FULL model,
    including nuisance regressors. This is not a scalar residual-only bootstrap.
    A QR basis avoids singular dummy columns and records the complete rank.
    """
    y, x = np.asarray(y, float), np.asarray(x, float)
    q = orthogonal_basis(controls)
    xr = x - q @ (q.T @ x)
    xx = float(xr @ xr)
    if xx < 1e-12:
        raise ValueError("no residual target variation")
    n, k = len(y), q.shape[1] + 1
    ids, levels = pd.factorize(pd.Series(clusters).astype(str), sort=True)
    g = len(levels)
    if g < 2 or n <= k:
        raise ValueError("insufficient residual degrees of freedom")
    beta = float(xr @ y / xx)
    u0 = y - q @ (q.T @ y)
    u = u0 - beta * xr
    influence = xr / xx
    scores = np.bincount(ids, weights=influence * u, minlength=g)
    ssc = g / (g - 1) * (n - 1) / (n - k)
    se = float(np.sqrt(ssc * (scores @ scores)))
    t = beta / se if se > 0 else np.nan
    qfull = np.column_stack([q, xr / np.sqrt(xx)])
    target_share = np.bincount(ids, weights=xr * xr, minlength=g) / xx
    critical = stats.t.ppf(.975, g-1)
    row = dict(estimate=beta, se_cluster=se, ci_low=beta - critical * se, ci_high=beta + critical * se,
               p_cluster=float(2 * stats.t.sf(abs(t), g - 1)), p_wild=np.nan,
               n_clusters=g, model_rank=k, residual_df=n-k, target_ss=xx,
               effective_identifying_clusters=float(1 / (target_share @ target_share)),
               max_identifying_cluster_share=float(target_share.max()),
               clusters_with_target_variation=int((target_share > 1e-10).sum()),
               bootstrap_draws=boot, bootstrap_seed=seed, bootstrap_type="WCR11", bootstrap_weights="Rademacher", bootstrap_impose_null=True,
               interval_reference="t_G_minus_1", variance="CRV1_full_model_rank")
    if boot:
        a = np.bincount(ids, weights=influence * u0, minlength=g)
        left = np.zeros((g, k))
        right = np.zeros((g, k))
        np.add.at(left, ids, influence[:, None] * qfull)
        np.add.at(right, ids, u0[:, None] * qfull)
        matrix = left @ right.T
        weights, actual_b = draw_weights("rademacher", g < 20 and 2**g < boot, g, boot, np.random.default_rng(seed))
        ts = []
        for start in range(0, actual_b, 512):
            v = weights[:, start:start+512]
            numer = a @ v
            bs = a[:, None] * v - matrix @ v
            denom = np.sqrt(ssc * np.sum(bs * bs, axis=0))
            ts.extend(np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 1e-14))
        ts = np.asarray(ts)
        if not np.isfinite(ts).all():
            raise ValueError("nonfinite bootstrap studentization")
        row.update(p_wild=float(np.mean(np.abs(ts) > abs(t))), bootstrap_draws=actual_b)
    return row, pd.DataFrame({"cluster_1km": levels, "identifying_share": target_share})


def dummies(values, prefix):
    return pd.get_dummies(values.astype(str), prefix=prefix, drop_first=True, dtype=float)


def controls_for(d, specification="primary", extra=None):
    c = pd.DataFrame({"intercept": 1.0, "inside_main": d.inside.astype(float),
                      "sfha_post": d.inside_sfha.astype(float) * d.post}, index=d.index)
    c = pd.concat([c, dummies(d.calendar_month, "month"), dummies(d.cluster_1km, "grid")], axis=1)
    if specification == "sfha_by_event_year":
        for year in sorted(d.event_year.unique()):
            c[f"sfha_year_{year}"] = d.inside_sfha.astype(float) * d.event_year.eq(year)
    if specification == "neighborhood_by_event_year":
        c = pd.concat([c, dummies(d.neighborhood_id.astype(str) + "_" + d.event_year.astype(str), "neighborhood_year")], axis=1)
    if extra is not None:
        c = pd.concat([c, extra], axis=1)
    return c


def outcome_sample(data, outcome):
    d = data.copy()
    if outcome == "zero_improvement_share":
        d = d[d.improvement_category.ne("unknown")].copy()
        d["outcome_value"] = d.improvement_category.eq("zero").astype(float)
    elif outcome in ("pooled_log_price", "improved_log_price", "zero_log_price"):
        d = d[d.price_eligible].copy()
        if outcome != "pooled_log_price":
            d = d[d.improvement_category.eq("positive" if outcome == "improved_log_price" else "zero")].copy()
        d["outcome_value"] = d.log_price_real
    else:
        d = d[d.final_class.notna()].copy()
        d["outcome_value"] = (d.final_class.eq("LLC") if outcome == "llc_share" else d.final_class.ne("Individual")).astype(float)
    return d.reset_index(drop=True)


def model(data, outcome, specification="primary", boot=9999, target=None, extra=None):
    d = outcome_sample(data, outcome)
    if specification == "neighborhood_by_event_year":
        d = d[d.neighborhood_id.notna()].reset_index(drop=True)
    row = {"outcome": outcome, "specification": specification, **support(d)}
    if not row["support_gate"]:
        return {**row, "status": "insufficient_sample_support", "estimate": np.nan, "p_wild": np.nan}, pd.DataFrame()
    x = d.inside.to_numpy(float) * d.post.to_numpy(float) if target is None else target(d)
    try:
        fit, concentration = full_model_fit(d.outcome_value, x, controls_for(d, specification, None if extra is None else extra(d)), d.cluster_1km, boot)
    except ValueError as e:
        return {**row, "status": str(e), "estimate": np.nan, "p_wild": np.nan}, pd.DataFrame()
    concentration["outcome"] = outcome
    concentration["specification"] = specification
    return {**row, **fit, "status": "estimated"}, concentration


def geographic_sample(d, comparison="outside_0_1000"):
    lo, hi = {"outside_0_1000": (0, 1000), "outside_0_300": (0, 300), "outside_1000_3000": (1000, 3000)}[comparison]
    outside = d.inside.eq(0) & d.signed_dist_inund_m.ge(lo) & d.signed_dist_inund_m.le(hi)
    if lo:
        outside &= d.signed_dist_inund_m.gt(lo)
    return d[d.inside.eq(1) | outside].copy().reset_index(drop=True)


def load_data(root, out):
    files = [root / "data_work" / x for x in ["transactions_linked.parquet", "parcel_boundary_distances.parquet", "parcel_centroids.gpkg", "assessor_raw.parquet", "panel_parcel_month.parquet", "restricted/grantee_classification/grantee_model_labels.parquet"]]
    raw_path = root / "data_work/transactions_raw.parquet"
    raw = pd.read_parquet(raw_path)
    raw["sale_date"] = pd.to_datetime(raw.sale_date, errors="coerce")
    raw = bundle_flags(raw)
    ledger = pd.read_parquet(files[0])
    ledger["parcel_id"] = ledger.parcel_id.astype("string")
    ledger["sale_date"] = pd.to_datetime(ledger.sale_date, errors="coerce")
    flags = ["instrument_key", "instrument_parcels", "instrument_dates", "instrument_parcel_prices", "repeated_party_amount_parcels", "ambiguous_bundle", "duplicate_instrument_parcel"]
    ledger = ledger.merge(raw[["sale_id"]+flags], on="sale_id", validate="one_to_one", how="left")
    if ledger[flags].isna().any().any():
        raise ValueError("linked deed missing full-ledger instrument audit")
    flow = []
    def step(name, f):
        record = {"stage": name, "records": len(f), "parcels": f.parcel_id.nunique()}
        if "inside" in f:
            record.update(support(f))
        flow.append(record)
        return f
    step("00_all_delivered_records", raw)
    d = step("01_linked_delivery", ledger)
    d = step("02_high_confidence_identifiers", d[d.parcel_match_method.isin(["location_id_direct", "location_id_zero_unpad"])])
    d = step("03_valid_dated_rows", d[d.input_row_valid.eq(True) & d.sale_date.notna()])
    d = step("04_exact_study_period", d[d.sale_date.ge("2015-03-13") & d.sale_date.lt("2022-03-13")])
    d = step("05_residential_at_sale", d[code(d.sale_property_type).isin([1, 6, 7])])
    d = step("06_arms_length", d[code(d.sale_usability_code).isin([0, 1, 2, 3, 5])])
    d = step("07_unique_instrument_parcel", d[~d.duplicate_instrument_parcel]).copy()
    dist = pd.read_parquet(files[1])
    dist["parcel_id"] = dist.parcel_id.astype("string")
    d = step("08_saved_exposure_link", d.merge(dist, on="parcel_id", how="inner", validate="many_to_one"))
    c = gpd.read_file(files[2], layer="parcel_centroids").to_crs(32615)
    c["parcel_id"] = c.parcel_id.astype("string")
    c["utm_x"], c["utm_y"] = c.geometry.x, c.geometry.y
    c["cluster_1km"] = np.floor(c.utm_x/1000).astype(int).astype(str) + "_" + np.floor(c.utm_y/1000).astype(int).astype(str)
    a = pd.read_parquet(files[3], columns=["Parcel_ID", "Neighborhood", "Subdivision", "Parcel_URL"])
    a["parcel_id"] = a.Parcel_ID.astype("string")
    dup = a[a.parcel_id.duplicated(False)]
    if len(dup) and dup.groupby("parcel_id")[["Neighborhood", "Subdivision"]].nunique().gt(1).any().any():
        raise ValueError("conflicting assessor attributes for a repeated parcel identifier")
    a = a.drop_duplicates("parcel_id").rename(columns={"Neighborhood": "neighborhood_id", "Subdivision": "subdivision"})
    d = d.merge(c[["parcel_id", "utm_x", "utm_y", "cluster_1km"]], on="parcel_id", how="left", validate="many_to_one")
    d = d.merge(a[["parcel_id", "neighborhood_id", "subdivision", "Parcel_URL"]], on="parcel_id", how="left", validate="many_to_one")
    if d.cluster_1km.isna().any():
        raise ValueError("missing spatial cluster")
    d["subdivision"] = d.subdivision.astype("string").fillna("UNRECORDED").str.strip().str.upper().replace("", "UNRECORDED")
    # The assessor's Subdivision field is a code, not a place name.
    prefixes = d.legal_desc.astype("string").str.upper().str.split(r"\s+(?:LOT|BLOCK|BLK|SEC-TWN-RGE)\s+", n=1, regex=True).str[0].str.strip()
    name_source = pd.DataFrame({"subdivision": d.subdivision, "label": prefixes}).dropna()
    names = name_source.groupby("subdivision").label.agg(lambda x: x.value_counts().index[0]).to_dict()
    names["60000"] = "LANDS (unplatted tracts)"
    d["subdivision_label"] = d.subdivision.map(names).fillna(d.subdivision)
    d["inside"] = d.inside_inund.astype(int)
    d["event_year"] = event_year(d.sale_date)
    d["post"] = d.event_year.ge(0).astype(int)
    d["calendar_month"] = d.sale_date.dt.to_period("M").astype(str)
    d["improvement_category"] = improvement_category(d.sale_improvement_value)
    d["price_eligible"] = d.price_nominal.gt(1000) & np.isfinite(d.price_nominal) & ~d.ambiguous_bundle
    cpi_path = root / "doc/provenance/flood_market_cpi_2015_2022.csv"
    cpi = pd.read_csv(cpi_path).set_index("year").cpi.to_dict()
    if set(cpi) != set(range(2015, 2023)) or any(not np.isfinite(x) or x<=0 for x in cpi.values()):
        raise ValueError("incomplete or invalid BLS annual CPI series")
    d["price_real"] = d.price_nominal * d.sale_date.dt.year.map(lambda yr: cpi[2022]/cpi[yr])
    d["log_price_real"] = np.log(d.price_real.where(d.price_real.gt(0)))
    d["building_complete"] = pd.to_numeric(d.floor_area, errors="coerce").gt(0) & pd.to_numeric(d.construction_year, errors="coerce").gt(0) & pd.to_numeric(d.construction_year, errors="coerce").le(d.sale_date.dt.year)
    old = pd.read_parquet(files[4], columns=["parcel_id"]).parcel_id.astype("string")
    d["old_cohort"] = d.parcel_id.isin(old)
    label_module_path = root / "src/02_labels/deed_grantee_classification.py"
    spec = importlib.util.spec_from_file_location("ff_grantee", label_module_path)
    label_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(label_module)
    d["grantee_key"] = label_module.normalize_name(d.grantee).map(label_module.grantee_key)
    labels = pd.read_parquet(files[5])
    extended_path = out / "restricted/grantee_expanded_labels.parquet"
    model_source_paths = [label_module.DEFAULT_TRAINING, label_module.DEFAULT_PUBLISHED_PREDICTIONS]
    target_mask = d.inside.eq(1) | d.signed_dist_inund_m.between(0, 3000)
    normalized = label_module.normalize_name(d.grantee)
    target = pd.DataFrame({"grantee_key": d.grantee_key, "name_group": normalized, "grantee_name": d.grantee}).loc[target_mask & normalized.ne("")].drop_duplicates("grantee_key").sort_values("grantee_key")
    missing = target[~target.grantee_key.isin(labels.grantee_key)]
    if len(missing):
        print(f"Applying the existing grantee classifier to {len(missing)} newly included names", flush=True)
        training = label_module.load_training(model_source_paths[0])
        published, _ = label_module.load_published_predictions(model_source_paths[1])
        classifier = label_module.build_distilled_model()
        classifier.fit(pd.concat([published.owner_name, training.owner_name], ignore_index=True), pd.concat([published.published_class, training.label], ignore_index=True))
        extra = label_module.classify_targets(missing, published, classifier)
        extra = extra[labels.columns]
        labels = pd.concat([labels, extra], ignore_index=True)
        extended_path.parent.mkdir(exist_ok=True)
        labels.to_parquet(extended_path, index=False)
    d = d.merge(labels[["grantee_key", "final_class", "classification_source"]], on="grantee_key", how="left", validate="many_to_one")
    primary = geographic_sample(d)
    step("09_full_footprint_plus_outside_1km", primary)
    step("diagnostic_old_cohort_only", primary[primary.old_cohort])
    step("diagnostic_old_cohort_and_symmetric_300m", primary[primary.old_cohort & primary.signed_dist_inund_m.abs().le(300)])
    step("diagnostic_symmetric_300m_without_old_cohort", primary[primary.signed_dist_inund_m.abs().le(300)])
    step("diagnostic_price_floor_only", primary[primary.price_nominal.gt(1000) & np.isfinite(primary.price_nominal)])
    step("10_price_eligible", primary[primary.price_eligible])
    step("diagnostic_ratio_eligible_prices", primary[primary.price_eligible & primary.ratio_study_default_usable])
    step("diagnostic_complete_building_prices", primary[primary.price_eligible & primary.building_complete])
    step("diagnostic_ratio_and_complete_building_prices", primary[primary.price_eligible & primary.ratio_study_default_usable & primary.building_complete])
    pd.DataFrame(flow).to_csv(out / "sample_flow.csv", index=False)
    restricted = out / "restricted"
    restricted.mkdir(exist_ok=True)
    d.to_parquet(restricted / "eligible_deeds.parquet", index=False)
    instrument = raw[raw.ambiguous_bundle | raw.duplicate_instrument_parcel]
    instrument.to_parquet(restricted / "instrument_audit.parquet", index=False)
    strata = list(primary.sort_values("sale_id").groupby(["inside", "post", "improvement_category"]))
    selected = []
    # Allocate the 40-record cap evenly across observed strata, then redistribute.
    pools = [z.sample(frac=1, random_state=SEED) for _, z in strata]
    for position in range(40):
        pool = position % len(pools)
        rank = position // len(pools)
        if rank < len(pools[pool]):
            selected.append(pools[pool].iloc[[rank]])
    sample = pd.concat(selected, ignore_index=True)
    sample["public_record_review_status"] = "pending"
    sample.to_csv(restricted / "measurement_review_records.csv", index=False)
    summary = []
    for flag in flags[1:]:
        summary.append({"check": flag, "all_raw_flagged": int((raw[flag].gt(1) if raw[flag].dtype != bool else raw[flag]).sum()),
                        "primary_flagged": int((primary[flag].gt(1) if primary[flag].dtype != bool else primary[flag]).sum())})
    pd.DataFrame(summary).to_csv(out / "instrument_checks.csv", index=False)
    return d, primary, files + [raw_path, label_module_path, cpi_path] + model_source_paths


def measurement_outputs(d, out):
    values = []
    for (inside, post, category), z in d.groupby(["inside", "post", "improvement_category"]):
        values.append(dict(inside=inside, post=post, category=category, deeds=len(z),
                           at_sale_equals_current=int(z.sale_improvement_value.eq(z.current_improvement_value).sum()),
                           current_improvement_positive=int(z.current_improvement_value.gt(0).sum()),
                           future_build_year=int(pd.to_numeric(z.construction_year, errors="coerce").gt(z.sale_date.dt.year).sum()),
                           floor_area_missing_or_zero=int((~pd.to_numeric(z.floor_area, errors="coerce").gt(0)).sum()),
                           code3=int(code(z.sale_usability_code).eq(3).sum()),
                           ambiguous_bundle=int(z.ambiguous_bundle.sum())))
    pd.DataFrame(values).to_csv(out / "measurement_summary.csv", index=False)
    d.groupby(["inside", "post", "sale_usability_code", "improvement_category"], dropna=False).size().rename("deeds").reset_index().to_csv(out / "sale_code_composition.csv", index=False)
    annual = d.groupby(["inside", "event_year", "improvement_category"]).agg(deeds=("sale_id", "size"), parcels=("parcel_id", "nunique"), median_nominal_price=("price_nominal", "median")).reset_index()
    annual["share"] = annual.deeds / annual.groupby(["inside", "event_year"]).deeds.transform("sum")
    annual.to_csv(out / "annual_composition.csv", index=False)
    d[d.price_eligible].groupby(["inside", "event_year", "improvement_category"]).agg(deeds=("sale_id", "size"), mean_log_price=("log_price_real", "mean"), median_real_price=("price_real", "median")).reset_index().to_csv(out / "annual_prices.csv", index=False)
    d[d.price_eligible].groupby(["inside", "post", "improvement_category"]).agg(deeds=("sale_id", "size"), mean_log_price=("log_price_real", "mean"), median_real_price=("price_real", "median")).reset_index().to_csv(out / "decomposition_cells.csv", index=False)
    amounts = []
    for (inside, post), z in d.groupby(["inside", "post"]):
        expected = z.total_consideration-z.non_real_consideration
        amounts.append(dict(inside=inside, post=post, deeds=len(z), sale_amount_equals_total_less_nonreal=int(np.isclose(z.price_nominal, expected, equal_nan=False).sum()), nonzero_assessor_adjustment=int(z.assessor_adjustment.fillna(0).ne(0).sum()), code2=int(code(z.sale_usability_code).eq(2).sum())))
    pd.DataFrame(amounts).to_csv(out / "price_measurement_checks.csv", index=False)
    d.groupby(["inside", "post", "classification_source"], dropna=False).size().rename("deeds").reset_index().to_csv(out / "grantee_classification_coverage.csv", index=False)
    for geo in ["cluster_1km", "subdivision"]:
        tab = d[d.inside.eq(1)].groupby(geo).agg(deeds=("sale_id", "size"), parcels=("parcel_id", "nunique"), pre=("post", lambda x: int(x.eq(0).sum())), post=("post", "sum")).reset_index().sort_values("deeds", ascending=False)
        tab["share"] = tab.deeds / tab.deeds.sum()
        if geo == "subdivision":
            tab = tab.merge(d[["subdivision", "subdivision_label"]].drop_duplicates(), on="subdivision", validate="one_to_one")
        tab.to_csv(out / f"concentration_{geo}.csv", index=False)


def accounting_from_cells(count, total):
    """Arrays end in group(2), period(2), category(K). Exact pre-weight identity."""
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = total / count
        w = count / count.sum(axis=-1, keepdims=True)
        actual = total.sum(axis=-1) / count.sum(axis=-1)
        within = (w[..., :, 0, :] * (mu[..., :, 1, :] - mu[..., :, 0, :])).sum(axis=-1)
        mix = ((w[..., :, 1, :] - w[..., :, 0, :]) * mu[..., :, 1, :]).sum(axis=-1)
        change = actual[..., :, 1] - actual[..., :, 0]
        return np.stack([change[..., 1] - change[..., 0], within[..., 1] - within[..., 0], mix[..., 1] - mix[..., 0]], axis=-1)


def decomposition(data, boot):
    d = data[data.price_eligible].copy()
    cats = [x for x in CATEGORIES if d.improvement_category.eq(x).any()]
    groups = sorted(d.cluster_1km.unique())
    count = np.zeros((len(groups), 2, 2, len(cats)))
    total = np.zeros_like(count)
    for (cl, inside, post, cat), z in d.groupby(["cluster_1km", "inside", "post", "improvement_category"]):
        idx = groups.index(cl), int(inside), int(post), cats.index(cat)
        count[idx], total[idx] = len(z), z.log_price_real.sum()
    est = accounting_from_cells(count.sum(axis=0), total.sum(axis=0))
    if not np.isfinite(est).all():
        return pd.DataFrame([{"status": "category_missing_in_group_period"}])
    assert abs(est[0]-est[1]-est[2]) < 1e-10
    draws = []
    rng = np.random.default_rng(SEED)
    for start in range(0, boot, 500):
        w = rng.multinomial(len(groups), np.ones(len(groups))/len(groups), size=min(500, boot-start))
        cc = (w @ count.reshape(len(groups), -1)).reshape(-1, 2, 2, len(cats))
        tt = (w @ total.reshape(len(groups), -1)).reshape(-1, 2, 2, len(cats))
        draws.append(accounting_from_cells(cc, tt))
    b = np.concatenate(draws) if draws else np.empty((0, 3))
    valid = np.isfinite(b).all(axis=1)
    rows = []
    for j, name in enumerate(["observed_relative_change", "fixed_preperiod_composition_change", "composition_contribution"]):
        ci = np.quantile(b[valid, j], [0.025, 0.975]) if valid.sum() and valid.mean() >= .95 else [np.nan, np.nan]
        rows.append(dict(component=name, estimate=est[j], ci_low=ci[0], ci_high=ci[1], bootstrap_draws=boot, valid_draws=int(valid.sum()), invalid_draws=int((~valid).sum()), bootstrap_seed=SEED, status="accounting_identity_not_causal", n_obs=len(d)))
    return pd.DataFrame(rows)


def event_models(d, outcome, boot=9999):
    """Joint CRV1 annual model; no coefficient when outcome support fails."""
    z = outcome_sample(d, outcome)
    gate = support(z)
    if not gate["support_gate"]:
        return pd.DataFrame([{**gate, "outcome": outcome, "status": "insufficient_sample_support"}])
    years = [-4, -3, -2, 0, 1, 2]
    c = controls_for(z).drop(columns=["sfha_post"])
    for yr in years:
        c[f"sfha_y{yr}"] = z.inside_sfha * z.event_year.eq(yr)
    q = orthogonal_basis(c)
    xx = np.column_stack([(z.inside * z.event_year.eq(yr)).to_numpy(float) for yr in years])
    xr = xx - q @ (q.T @ xx)
    yr = z.outcome_value.to_numpy(float) - q @ (q.T @ z.outcome_value.to_numpy(float))
    rank = np.linalg.matrix_rank(xr)
    if rank < len(years):
        return pd.DataFrame([{**gate, "outcome": outcome, "status": "annual_target_rank_failure"}])
    bread = np.linalg.inv(xr.T @ xr)
    beta = bread @ xr.T @ yr
    u = yr-xr@beta
    codes, ids = pd.factorize(z.cluster_1km, sort=True)
    sc = np.zeros((len(ids), len(years)))
    np.add.at(sc, codes, xr * u[:, None])
    k = q.shape[1]+len(years)
    vc = len(ids)/(len(ids)-1)*(len(z)-1)/(len(z)-k)*bread@sc.T@sc@bread
    pre = beta[:3]
    w = float(pre @ np.linalg.pinv(vc[:3, :3]) @ pre / 3)
    joint = float(stats.f.sf(w, 3, len(ids)-1))
    rows = []
    for i, ey in enumerate(years):
        cells = z[z.event_year.isin([-1, ey])].copy()
        cells["post"] = cells.event_year.eq(ey).astype(int)
        annual_support = support(cells)
        if not annual_support["support_gate"]:
            rows.append({**gate, "outcome": outcome, "event_year": ey, "estimate": np.nan,
                         "annual_reference_min_cell": annual_support["min_cell"], "annual_reference_four_cell_clusters": annual_support["four_cell_clusters"],
                         "status": "insufficient_annual_reference_support"})
            continue
        full_controls = np.column_stack([c.to_numpy(float), np.delete(xx, i, axis=1)])
        fit, _ = full_model_fit(z.outcome_value, xx[:, i], full_controls, z.cluster_1km, boot)
        if not np.isclose(fit["estimate"], beta[i], atol=1e-9):
            raise AssertionError("annual full-model and joint coefficients disagree")
        rows.append({**gate, **fit, "outcome": outcome, "event_year": ey, "joint_preperiod_p_crv1": joint,
                     "annual_reference_min_cell": annual_support["min_cell"], "annual_reference_four_cell_clusters": annual_support["four_cell_clusters"],
                     "status": "estimated_full_annual_model_descriptive"})
    return pd.DataFrame(rows)


def repeat_sales(d, boot):
    z = d[d.price_eligible].sort_values(["sale_date", "sale_id"])
    pre = z[z.post.eq(0)].groupby("parcel_id").tail(1)
    post = z[z.post.eq(1)].groupby("parcel_id").head(1)
    cols = ["parcel_id", "sale_date", "calendar_month", "log_price_real", "improvement_category", "inside", "inside_sfha", "cluster_1km"]
    pairs = pre[cols].merge(post[cols], on="parcel_id", validate="one_to_one", suffixes=("_pre", "_post"))
    if not (pairs.inside_pre.eq(pairs.inside_post) & pairs.cluster_1km_pre.eq(pairs.cluster_1km_post)).all():
        raise ValueError("repeat-pair exposure or cluster changed")
    pairs["transition"] = pairs.improvement_category_pre + "_to_" + pairs.improvement_category_post
    transitions = pairs.groupby(["inside_pre", "transition"]).size().rename("pairs").reset_index()
    rows = []
    for label, sample in [("all_pairs", pairs), ("positive_to_positive", pairs[pairs.transition.eq("positive_to_positive")])]:
        inside = sample[sample.inside_pre.eq(1)]
        gate = len(inside) >= 25 and inside.cluster_1km_pre.nunique() >= 10
        row = dict(sample=label, n_pairs=len(sample), inside_pairs=len(inside), inside_clusters=inside.cluster_1km_pre.nunique(), support_gate=gate)
        if not gate:
            rows.append({**row, "status": "insufficient_repeat_support", "estimate": np.nan})
            continue
        sample = sample.reset_index(drop=True)
        periods = sorted(set(sample.calendar_month_pre) | set(sample.calendar_month_post))
        c = pd.DataFrame({"intercept": np.ones(len(sample)), "sfha": sample.inside_sfha_pre.astype(float)})
        for month in periods[1:]:
            c[f"month_{month}"] = sample.calendar_month_post.eq(month).astype(float)-sample.calendar_month_pre.eq(month).astype(float)
        try:
            fit, _ = full_model_fit(sample.log_price_real_post-sample.log_price_real_pre, sample.inside_pre, c, sample.cluster_1km_pre, boot)
            rows.append({**row, **fit, "status": "estimated"})
        except ValueError as e:
            rows.append({**row, "status": str(e), "estimate": np.nan})
    return pd.DataFrame(rows), transitions


def run(project_root=None, output_dir=None, boot=9999, skip_leaveouts=False):
    root = Path(project_root) if project_root else ROOT
    out = Path(output_dir) if output_dir else root / "data_work/flood_market_composition"
    plan = root / "docs/METHODS.md"
    if not plan.exists():
        raise RuntimeError("Record the authorized analysis plan before estimation")
    if boot < 1:
        raise ValueError("positive bootstrap draw count required; manuscript exporter requires 9999")
    out.mkdir(parents=True, exist_ok=True)
    # A rerun invalidates prior validation; do not allow stale checks into its manifest.
    for name in ("validation_checks.csv", "full_model_reference_validation.csv"):
        (out / name).unlink(missing_ok=True)
    start = datetime.now(timezone.utc).isoformat()
    files_before = sorted((root / "manuscript_quarto/data").glob("transaction_*.csv"))
    old_hashes = {str(p): sha256(p) for p in files_before if not p.name.startswith("._")}
    all_data, d, inputs = load_data(root, out)
    measurement_outputs(d, out)
    rows, concentrations = [], []
    for outcome in OUTCOMES:
        print(f"Primary {outcome}", flush=True)
        result, conc = model(d, outcome, boot=boot)
        rows.append(result)
        concentrations.append(conc)
    primary = pd.DataFrame(rows)
    primary["p_holm"] = np.nan
    ok = primary.p_wild.notna()
    # Include unsupported outcomes as p=1 in the fixed three-test family.
    adjusted = multipletests(primary.p_wild.fillna(1).to_numpy(), method="holm")[1]
    primary.loc[ok, "p_holm"] = adjusted[ok]
    primary.to_csv(out / "primary_estimates.csv", index=False)
    pd.concat(concentrations, ignore_index=True).to_csv(out / "identifying_concentration.csv", index=False)
    pd.concat([event_models(d, o, boot) for o in OUTCOMES], ignore_index=True).to_csv(out / "event_study.csv", index=False)
    decomposition(d, boot).to_csv(out / "decomposition.csv", index=False)
    rep, transitions = repeat_sales(d, boot)
    rep.to_csv(out / "repeat_sales.csv", index=False)
    transitions.to_csv(out / "repeat_transitions.csv", index=False)
    secondary = [model(d, o, boot=boot)[0] for o in ("llc_share", "nonindividual_share", "zero_log_price")]
    for outcome in ["llc_share", "nonindividual_share"]:
        secondary.append(model(d[d.classification_source.eq("published_bert_production_cache_exact_name")], outcome, specification="stored_bert_only", boot=boot)[0])
    pd.DataFrame(secondary).to_csv(out / "secondary_outcomes.csv", index=False)
    sensitivities = []
    for outcome in OUTCOMES:
        for spec in ["outside_0_300", "outside_1000_3000", "sfha_by_event_year", "neighborhood_by_event_year", "ratio_eligible", "building_complete", "ratio_and_building_complete", "no_bundle_all_outcomes", "exclude_code2"]:
            print(f"Sensitivity {outcome}: {spec}", flush=True)
            z = geographic_sample(all_data, spec) if spec.startswith("outside_") else d.copy()
            if spec in ["ratio_eligible", "ratio_and_building_complete"]:
                z = z[z.ratio_study_default_usable]
            if spec in ["building_complete", "ratio_and_building_complete"]:
                z = z[z.building_complete]
            if spec == "no_bundle_all_outcomes":
                z = z[~z.ambiguous_bundle]
            if spec == "exclude_code2":
                z = z[~code(z.sale_usability_code).eq(2)]
            result, _ = model(z, outcome, specification=spec, boot=boot)
            base = primary.loc[primary.outcome.eq(outcome)].iloc[0]
            result["observations_lost_vs_primary"] = int(base.n_obs-result["n_obs"])
            result["target_ss_relative_to_primary"] = result.get("target_ss", np.nan)/base.get("target_ss", np.nan)
            sensitivities.append(result)
        for label, mask, cutoff in [("first_post_year", d.event_year.le(0), "2019-03-13"), ("later_post_years", d.event_year.ne(0), "2019-03-13"), ("placebo_2017_pre_event_only", d.sale_date.lt("2019-03-13"), "2017-03-13")]:
            z = d[mask].copy()
            z["post"] = z.sale_date.ge(cutoff).astype(int)
            result, _ = model(z, outcome, specification=label, boot=boot)
            sensitivities.append(result)
    pd.DataFrame(sensitivities).to_csv(out / "sensitivity_estimates.csv", index=False)
    leaveouts = []
    if not skip_leaveouts:
        for geography in ["cluster_1km", "subdivision"]:
            for location in sorted(d.loc[d.inside.eq(1), geography].unique()):
                z = d[d[geography].ne(location)]
                for outcome in OUTCOMES:
                    row, _ = model(z, outcome, specification="leave_one_out", boot=boot)
                    leaveouts.append({**row, "geography": geography, "omitted_location": location, "removed_records": len(d)-len(z)})
            print(f"Completed leave-one-{geography}-out", flush=True)
    pd.DataFrame(leaveouts).to_csv(out / "leave_one_out.csv", index=False)
    benchmarks = []
    for _, row in primary.iterrows():
        for margin in ([.05, .10] if row.outcome == "zero_improvement_share" else [np.log(1.05), np.log(1.10)]):
            low = -margin if row.outcome == "zero_improvement_share" else np.log(2-np.exp(margin))
            benchmarks.append({"outcome": row.outcome, "lower_benchmark": low, "upper_benchmark": margin,
                               "interval_inside_benchmarks": bool(row.get("ci_low", -np.inf)>low and row.get("ci_high", np.inf)<margin), "interpretation": "CRV1 precision benchmark, not an equivalence test"})
    pd.DataFrame(benchmarks).to_csv(out / "precision_benchmarks.csv", index=False)
    if any(sha256(Path(p)) != h for p, h in old_hashes.items()):
        raise AssertionError("Previous article evidence changed")
    pd.DataFrame([{"path": p, "sha256": h, "unchanged": True} for p, h in old_hashes.items()]).to_csv(out / "preserved_benchmark_manifest.csv", index=False)
    metadata = dict(started_utc=start, completed_utc=datetime.now(timezone.utc).isoformat(), bootstrap_draws=boot, seed=SEED,
                    plan_sha256=sha256(plan), scope="residential_deeds_linked_to_2023_geography_full_inundation_footprint", primary_comparison="outside_0_1000m", prior_analysis_preserved=True,
                    currency="2022 dollars using BLS CUUR0200SA0 annual averages; calendar-month effects absorb calendar deflator differences", leaveouts_complete=not skip_leaveouts,
                    inferential_class="retrospective_descriptive_associations", central_measure="recorded_improvement_value_at_sale_not_verified_vacancy")
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    manifest = [{"role": "input", "path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in inputs+[plan, Path(__file__)]]
    manifest += [{"role": "restricted_output" if "restricted" in p.parts else "output", "path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.csv" and not p.name.startswith("._")]
    pd.DataFrame(manifest).to_csv(out / "manifest.csv", index=False)
    print(primary.to_string(index=False), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--boot", type=int, default=9999)
    ap.add_argument("--skip-leaveouts", action="store_true")
    args = ap.parse_args()
    run(args.project_root, args.output_dir, args.boot, args.skip_leaveouts)
