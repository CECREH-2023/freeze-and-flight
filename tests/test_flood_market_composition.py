"""Scientific invariants for the expanded-footprint analysis."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import unittest
import statsmodels.api as sm
from wildboottest.wildboottest import WildboottestCL

spec = importlib.util.spec_from_file_location("flood_market", Path(__file__).resolve().parents[1]/"src/07_estimation/flood_market_composition.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_exact_date_boundaries_and_residential_codes():
    dates = pd.Series(["2015-03-12", "2015-03-13", "2019-03-12", "2019-03-13", "2022-03-12", "2022-03-13"])
    assert m.event_year(dates).tolist() == [-5, -4, -1, 0, 2, 3]
    assert m.code(pd.Series(["01", "06", "07", "11", None])).isin([1, 6, 7]).tolist() == [True, True, True, False, False]


def test_missing_and_invalid_improvements_are_not_zero():
    assert m.improvement_category(pd.Series([1, 0, None, -1, np.inf, "bad", "0"])).tolist() == ["positive", "zero", "unknown", "unknown", "unknown", "unknown", "zero"]


def test_bundle_detection_precedes_filters_and_handles_conflicting_dates():
    d = pd.DataFrame({"sale_id": list("abcdef"), "sale_date": pd.to_datetime(["2018-01-01"]*5+["2018-01-02"]),
                      "deed_book": ["2018"]*6, "deed_page": ["1", "1", "2", "3", "4", "4"],
                      "parcel_id": ["A", "B", "C", "D", "E", "E"], "price_nominal": [30000, 30000, 10000, 10000, 20000, 20000],
                      "grantor": ["Seller"]*6, "grantee": ["Buyer"]*6})
    r = m.bundle_flags(d)
    assert r.ambiguous_bundle.all()
    assert r.instrument_parcels.iloc[0] == 2
    assert r.instrument_dates.iloc[-1] == 2
    assert r.repeated_party_amount_parcels.iloc[2] == 2
    duplicate = pd.concat([d, d.iloc[[0]].assign(sale_id="g")], ignore_index=True)
    assert m.bundle_flags(duplicate).duplicate_instrument_parcel.sum() == 1


def test_four_cells_must_be_in_each_counted_cluster():
    d = pd.DataFrame([(str(g), i, p) for g in range(10) for i in [0, 1] for p in [0, 1] for _ in range(2)], columns=["cluster_1km", "inside", "post"])
    assert m.support(d)["support_gate"]
    d = d[~((d.cluster_1km == "0") & (d.inside == 1) & (d.post == 1))]
    r = m.support(d)
    assert r["four_cell_clusters"] == 9 and not r["support_gate"]
    # Increasing the three remaining cells cannot fix missing four-cell support.
    assert not m.support(pd.concat([d]*3))["support_gate"]


def test_geographic_comparisons_keep_entire_footprint():
    d = pd.DataFrame({"inside": [1, 1, 0, 0, 0, 0, 0], "signed_dist_inund_m": [-5000, -1, 0, 300, 1000, 1000.1, 3000]})
    for label, expected in [("outside_0_1000", [-5000, -1, 0, 300, 1000]), ("outside_0_300", [-5000, -1, 0, 300]), ("outside_1000_3000", [-5000, -1, 1000.1, 3000])]:
        assert m.geographic_sample(d, label).signed_dist_inund_m.tolist() == expected


def test_exact_pre_share_decomposition_and_no_mix_case():
    count = np.array([[[20, 80], [50, 50]], [[70, 30], [30, 70]]], dtype=float)
    mu = np.array([[[10, 12], [11, 13]], [[9, 12], [10, 14]]], dtype=float)
    r = m.accounting_from_cells(count, count*mu)
    np.testing.assert_allclose(r[0], r[1]+r[2], atol=1e-12)
    count[:, 1, :] = count[:, 0, :]
    r = m.accounting_from_cells(count, count*mu)
    np.testing.assert_allclose(r[2], 0, atol=1e-12)


def reference_comparison(seed):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(16), 18)
    n = len(groups)
    z = rng.normal(size=(n, 3))
    controls = np.column_stack([np.ones(n), z, pd.get_dummies(groups, drop_first=True).to_numpy(float)])
    x = rng.normal(size=n) + z[:, 0] + .2*groups
    y = .08*x + z@np.array([1, 2, -.5]) + rng.normal(size=16)[groups] + rng.normal(size=n)
    fit, shares = m.full_model_fit(y, x, controls, groups, boot=499, seed=seed)
    X = np.column_stack([controls, x])
    ref = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True})
    np.testing.assert_allclose(fit["estimate"], ref.params[-1], atol=1e-10)
    np.testing.assert_allclose(fit["se_cluster"], ref.bse[-1], atol=1e-10)
    assert fit["model_rank"] == X.shape[1]
    assert fit["residual_df"] == n-X.shape[1]
    np.testing.assert_allclose(shares.identifying_share.sum(), 1)
    R = np.zeros(X.shape[1]); R[-1] = 1
    canonical_groups = pd.factorize(pd.Series(groups).astype(str), sort=True)[0]
    wild = WildboottestCL(X, y, canonical_groups, R, B=499, seed=seed, parallel=False)
    wild.get_scores("11", impose_null=True, adj=True, cluster_adj=True)
    wild.get_weights("rademacher")
    wild.get_numer(); wild.get_denom(); wild.get_tboot(); wild.get_vcov(); wild.get_tstat(); wild.get_pvalue()
    np.testing.assert_allclose(fit["p_wild"], wild.pvalue, atol=1e-12)
    # Redundant nuisance columns may not artificially inflate rank corrections.
    redundant = m.full_model_fit(y, x, np.column_stack([controls, controls[:, 2]]), groups, boot=499, seed=seed)[0]
    np.testing.assert_allclose(redundant["se_cluster"], fit["se_cluster"], atol=1e-10)
    assert redundant["p_wild"] == fit["p_wild"]


def test_bootstrap_reproduces_and_refuses_absorbed_target():
    rng = np.random.default_rng(91)
    g = np.repeat(np.arange(12), 10)
    x, y = rng.normal(size=(2, 120))
    c = np.ones((120, 1))
    a = m.full_model_fit(y, x, c, g, boot=99)[0]
    b = m.full_model_fit(y, x, c, g, boot=99)[0]
    assert a["p_wild"] == b["p_wild"]
    with unittest.TestCase().assertRaisesRegex(ValueError, "no residual target variation"):
        m.full_model_fit(y, x, np.column_stack([c, x]), g, boot=99)


def test_price_gate_does_not_remove_unknown_categories_from_pooled_prices():
    d = pd.DataFrame({"price_eligible": [True]*3, "improvement_category": ["positive", "zero", "unknown"], "log_price_real": [11., 10., 12.]})
    assert len(m.outcome_sample(d, "pooled_log_price")) == 3
    assert len(m.outcome_sample(d, "improved_log_price")) == 1
    assert len(m.outcome_sample(d, "zero_improvement_share")) == 2


def test_reference_seed_3():
    reference_comparison(3)


def test_reference_seed_18():
    reference_comparison(18)


class FloodMarketTests(unittest.TestCase):
    pass


for name, function in list(globals().items()):
    if name.startswith("test_") and callable(function):
        setattr(FloodMarketTests, name, lambda self, f=function: f())
del name, function


if __name__ == "__main__":
    unittest.main()
