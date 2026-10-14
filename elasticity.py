"""
price_response.py

Price-response curves for rental listings, estimated from nightly history.

For each cohort (metro x vehicle_type) it estimates how the probability that an
available night gets booked changes with the listing's base price, then turns that
into occupancy and revenue-per-available-night (RevPAN) curves and a
revenue-maximizing price change.

Usage (notebook or script, with `history` already loaded):

    from price_response import run_price_response, save_elasticities
    results = run_price_response(history)       # a plain DataFrame, one row per cohort

    results.query("metro == 'Denver-Aurora-Lakewood, CO Metro Area' and not flag_thin_data")
    results.sort_values("revpan_lift_pct", ascending=False).head(20)
    save_elasticities(results, "elasticities.csv")   # monthly: append this version for recommend()

    # Run-level checks are in results.attrs["diagnostics"] (also as columns, see below).
    # Curves, holdout calibration and the fitted model, when you need them:
    results, details = run_price_response(history, return_details=True)
    details["curves"]        # DataFrame: occupancy / RevPAN by price multiplier, per cohort
    details["calibration"]   # DataFrame: holdout predicted vs observed, by decile

Columns of `results` (all plain str / int / float / bool, safe for .query() or a SQL table):
    elasticity_version   as_of date of the data (YYYY-MM-DD); one value per monthly run
    run_at               when the model ran
    cohort, metro, vehicle_type, vehicle_type_group
    listings, listings_changing_price, nights_all, nights_recent, occupancy_recent
    slope                cohort slope (logit change per 1.0 change in log price)
    slope_group_level    vehicle_type_group slope (the fallback)
    slope_overall        overall slope (last-resort fallback)
    slope_used           slope the recommender should use: cohort slope, or group slope if thin
    slope_source         "cohort" or "group"
    elasticity_now       % change in booked nights per 1% price change, at current prices
    optimal_multiplier, occupancy_at_optimal, revpan_now, revpan_at_optimal, revpan_lift_pct
    observed_range_low, observed_range_high   5th-95th pct of price moves seen in this cohort
    flag_thin_data, flag_nonnegative_slope, flag_optimum_at_grid_edge, flag_optimum_outside_range
    flags                the same flags as one readable string
    booking_rate_overall, share_listings_changing_price, naive_slope_overall

`history`: one row per listing per night. Required columns:
    calendar_date, listing_id, metro, vehicle_type, vehicle_type_group,
    is_weekend, bookable_days, booked_days, avg_base_price
Used as time-varying controls if present:
    review_count, total_review_score, image_count, description_length, instant_book

Model (in plain terms)
----------------------
Unit: listing x week x weekend/weekday, with booked nights out of available nights.
Booking probability is a logistic function of:
  * the listing's log base price relative to its cohort's median that week
    (this is the price-response term; its coefficient is the "slope")
  * a fixed effect per listing        -> absorbs quality: nicer listings price higher AND book more
  * a fixed effect per metro x week   -> absorbs demand: seasons, holidays, local events
  * weekend, and the as-of-day listing features (reviews, photos, etc.)
The slope is estimated per cohort but shrunk toward its vehicle_type_group, which is
shrunk toward the overall slope. Thin cohorts therefore borrow from broader ones,
the statistical version of a fallback chain.

What the curves mean: the response of ONE listing moving its price while the rest of
its market stays put. If every host in a cohort moved together, demand would respond
less than this, so treat the optimum as a per-listing recommendation, not a
market-wide one.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

CONFIG = dict(
    as_of=None,              # only stay dates before this date are used; None = today
    min_price=10.0,          # drop nights with base price below this (bad data)
    # Prior standard deviations, on the logit scale. Larger = less shrinkage.
    prior_sd_fixed=3.0,      # fixed effects, controls, overall slope: barely shrunk on the logit scale
    prior_sd_group=0.5,      # how far a vehicle_type_group slope may move from the overall slope
    prior_sd_cohort=0.5,     # how far a cohort slope may move from its group's slope
    price_grid=np.round(np.arange(0.70, 1.4001, 0.05), 2),  # multipliers on current base price
    curve_window_weeks=52,   # curves are built on the most recent N weeks of each cohort
    min_listings_changing=10,  # below this, a cohort's slope is marked low confidence
    min_nights=2000,
    evaluate=True,           # holdout check (adds two extra fits)
    holdout_frac=0.2,
    random_state=0,
    max_iter=3000,
    verbose=True,
)

REQUIRED = [
    "calendar_date", "listing_id", "metro", "vehicle_type", "vehicle_type_group",
    "is_weekend", "bookable_days", "booked_days", "avg_base_price",
]
OPTIONAL = ["review_count", "total_review_score", "image_count", "description_length", "instant_book"]


def _log(cfg, msg, t0=None):
    if cfg["verbose"]:
        suffix = f"  ({time.time() - t0:.1f}s)" if t0 is not None else ""
        print(f"[price_response] {msg}{suffix}", flush=True)


# --------------------------------------------------------------------------- data prep
def prepare_cells(history: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str], dict]:
    """Filter nightly history and aggregate to listing x week x weekend cells."""
    missing = [c for c in REQUIRED if c not in history.columns]
    if missing:
        raise ValueError(f"history is missing required columns: {missing}")
    opt = [c for c in OPTIONAL if c in history.columns]

    df = history[REQUIRED + opt].copy()
    n_raw = len(df)
    df["calendar_date"] = pd.to_datetime(df["calendar_date"])
    as_of = pd.Timestamp(cfg["as_of"]) if cfg["as_of"] else pd.Timestamp.today().normalize()

    # Past stay dates only: future nights haven't finished selling, so an unbooked
    # future night is not evidence of low demand.
    df = df[df["calendar_date"] < as_of]
    n_past = len(df)

    # Available = host had it open, or it was booked (covers either way a booked
    # night might be recorded in bookable_days).
    booked = df["booked_days"].fillna(0) > 0
    available = (df["bookable_days"].fillna(0) > 0) | booked
    price_ok = df["avg_base_price"].fillna(0) >= cfg["min_price"]
    df = df[available & price_ok]
    df["booked"] = booked[df.index].astype(np.int32)
    df["nights"] = np.int32(1)

    df["week"] = df["calendar_date"].dt.to_period("W-SUN").dt.start_time
    df["is_weekend"] = df["is_weekend"].fillna(0).astype(np.int8)
    df["log_price"] = np.log(df["avg_base_price"].astype(float))
    for c in ["metro", "vehicle_type", "vehicle_type_group"]:
        df[c] = df[c].astype(str)
    df["cohort"] = (df["metro"] + " | " + df["vehicle_type"]).astype("category")
    df["vehicle_type_group"] = df["vehicle_type_group"].astype("category")
    df["metro"] = df["metro"].astype("category")

    # As-of-day listing features. Listing fixed effects absorb anything constant, so these
    # only matter through changes over time (e.g. a host raising price after earning reviews).
    controls = []
    if "review_count" in df:
        rc = df["review_count"].fillna(0).clip(lower=0).astype(float)
        df["c_log_reviews"] = np.log1p(rc)
        controls.append("c_log_reviews")
        if "total_review_score" in df:
            score = df["total_review_score"].fillna(0).astype(float)
            df["c_avg_rating"] = np.where(rc > 0, score / rc.where(rc > 0, 1), 0.0)
            controls.append("c_avg_rating")
    if "image_count" in df:
        df["c_log_images"] = np.log1p(df["image_count"].fillna(0).clip(lower=0).astype(float))
        controls.append("c_log_images")
    if "description_length" in df:
        df["c_log_desc"] = np.log1p(df["description_length"].fillna(0).clip(lower=0).astype(float))
        controls.append("c_log_desc")
    if "instant_book" in df:
        ib = df["instant_book"].map({True: 1.0, False: 0.0, "true": 1.0, "false": 0.0,
                                     "True": 1.0, "False": 0.0, 1: 1.0, 0: 0.0})
        df["c_instant_book"] = ib.fillna(0.0)
        df["c_instant_book_missing"] = ib.isna().astype(float)
        controls += ["c_instant_book", "c_instant_book_missing"]

    # Cohort benchmark: median log base price over available nights, per cohort-week.
    df["log_bench"] = df.groupby(["cohort", "week"], observed=True)["log_price"].transform("median")

    keys = ["listing_id", "cohort", "vehicle_type_group", "metro", "week", "is_weekend"]
    agg = {"nights": "sum", "booked": "sum", "log_price": "mean", "log_bench": "mean"}
    agg.update({c: "mean" for c in controls})
    cells = df.groupby(keys, observed=True, sort=False).agg(agg).reset_index()
    cells["rel_price"] = cells["log_price"] - cells["log_bench"]
    cells = cells[cells["nights"] > 0].reset_index(drop=True)

    info = dict(
        rows_in=n_raw, rows_past=n_past, rows_available=len(df),
        as_of=str(as_of.date()),
        date_min=str(df["calendar_date"].min().date()) if len(df) else None,
        date_max=str(df["calendar_date"].max().date()) if len(df) else None,
        controls=controls,
    )
    return cells, controls, info


# --------------------------------------------------------------------------- design
def build_design(cells: pd.DataFrame, controls: list[str], cfg: dict, include_price: bool = True):
    """Sparse design matrix. Each column block is pre-scaled so that sklearn's single
    L2 penalty acts as a prior with the configured standard deviation."""
    n = len(cells)
    rows = np.arange(n)
    blocks, layout = [], {}
    col = 0

    def add(name, mat):
        nonlocal col
        blocks.append(mat.tocsr())
        layout[name] = (col, col + mat.shape[1])
        col += mat.shape[1]

    def onehot(codes, k, values=None, scale=1.0):
        data = np.full(n, scale) if values is None else np.asarray(values, float) * scale
        return sparse.csr_matrix((data, (rows, codes)), shape=(n, k))

    s_fix, s_grp, s_coh = cfg["prior_sd_fixed"], cfg["prior_sd_group"], cfg["prior_sd_cohort"]

    lst_codes, lst_uni = pd.factorize(cells["listing_id"])
    add("listing_fe", onehot(lst_codes, len(lst_uni), scale=s_fix))
    mw_codes, mw_uni = pd.factorize(
        cells["metro"].astype(str) + "|" + cells["week"].dt.strftime("%Y-%m-%d"))
    add("metro_week_fe", onehot(mw_codes, len(mw_uni), scale=s_fix))
    add("weekend", sparse.csr_matrix(cells[["is_weekend"]].to_numpy(float) * s_fix))

    if controls:
        c = cells[controls].to_numpy(float)
        mu, sd = c.mean(0), c.std(0)
        sd[sd == 0] = 1.0
        add("controls", sparse.csr_matrix((c - mu) / sd * s_fix))

    if include_price:
        rel = cells["rel_price"].to_numpy(float)
        g_codes = cells["vehicle_type_group"].cat.codes.to_numpy()
        c_codes = cells["cohort"].cat.codes.to_numpy()
        add("slope_global", sparse.csr_matrix(rel[:, None] * s_fix))
        add("slope_group", onehot(g_codes, len(cells["vehicle_type_group"].cat.categories), rel, s_grp))
        add("slope_cohort", onehot(c_codes, len(cells["cohort"].cat.categories), rel, s_coh))

    X = sparse.hstack(blocks, format="csr")
    scales = np.empty(X.shape[1])
    for name, (a, b) in layout.items():
        scales[a:b] = {"slope_group": s_grp, "slope_cohort": s_coh}.get(name, s_fix)
    return X, layout, scales


def fit_logit(X, booked, nights, cfg):
    """Binomial logit on aggregated counts: each cell enters as a 'booked' row and a
    'not booked' row, weighted by their night counts."""
    n = X.shape[0]
    y = np.r_[np.ones(n), np.zeros(n)]
    w = np.r_[booked, nights - booked].astype(float)
    keep = w > 0
    Xs = sparse.vstack([X, X], format="csr")[keep]
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=cfg["max_iter"], tol=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(Xs, y[keep], sample_weight=w[keep])
    if model.n_iter_[0] >= cfg["max_iter"]:
        print("[price_response] WARNING: solver hit max_iter; raise CONFIG['max_iter'].")
    return model


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _weighted_metrics(p, booked, nights):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = -(booked * np.log(p) + (nights - booked) * np.log(1 - p)).sum() / nights.sum()
    # Brier score at the night level: each night is 0/1 with predicted p.
    brier = (booked * (1 - p) ** 2 + (nights - booked) * p ** 2).sum() / nights.sum()
    return ll, brier


# --------------------------------------------------------------------------- main entry
def run_price_response(history: pd.DataFrame, config: dict | None = None,
                       return_details: bool = False):
    """Fit the price-response model. Returns a flat DataFrame, one row per cohort.
    With return_details=True, returns (DataFrame, dict of curves / calibration / model)."""
    cfg = {**CONFIG, **(config or {})}
    t0 = time.time()

    cells, controls, info = prepare_cells(history, cfg)
    _log(cfg, f"{info['rows_in']:,} rows in -> {info['rows_available']:,} available past nights "
              f"-> {len(cells):,} listing-week cells", t0)
    if len(cells) == 0:
        raise ValueError("No usable rows after filtering. Check as_of, bookable_days and avg_base_price.")

    booked = cells["booked"].to_numpy(float)
    nights = cells["nights"].to_numpy(float)

    # ---- diagnostics: how much price variation is there to learn from?
    lp = cells["log_price"].to_numpy()
    lst_mean = cells.assign(_w=nights * lp).groupby("listing_id")["_w"].transform("sum") / \
        cells.groupby("listing_id")["nights"].transform("sum")
    cells["own_dev"] = lp - lst_mean.to_numpy()          # deviation from listing's own average price
    rng = cells.groupby("listing_id")["log_price"].agg(lambda s: s.max() - s.min())
    changers = set(rng.index[rng > 0.01])
    cells["listing_changes_price"] = cells["listing_id"].isin(changers)

    diagnostics = dict(info)
    diagnostics.update(
        listings=int(cells["listing_id"].nunique()),
        cohorts=int(cells["cohort"].nunique()),
        booking_rate=float(booked.sum() / nights.sum()),
        share_listings_changing_price=float(len(changers) / max(1, cells["listing_id"].nunique())),
        share_nights_from_price_changers=float(nights[cells["listing_changes_price"]].sum() / nights.sum()),
    )

    # ---- naive slope (no fixed effects) for comparison: shows how much bias the FEs remove
    t1 = time.time()
    Xn = sparse.csr_matrix(np.c_[cells["rel_price"], cells["is_weekend"]].astype(float))
    naive = fit_logit(Xn, booked, nights, {**cfg, "max_iter": 500})
    diagnostics["naive_slope_no_fixed_effects"] = float(naive.coef_[0][0])
    _log(cfg, "naive model fit", t1)

    # ---- holdout check (random cells): does price improve fit, and is it calibrated?
    calibration = None
    X, layout, scales = build_design(cells, controls, cfg, include_price=True)
    if cfg["evaluate"]:
        t1 = time.time()
        rs = np.random.default_rng(cfg["random_state"])
        test = rs.random(len(cells)) < cfg["holdout_frac"]
        X0, _, _ = build_design(cells, controls, cfg, include_price=False)
        m_full = fit_logit(X[~test], booked[~test], nights[~test], cfg)
        m_base = fit_logit(X0[~test], booked[~test], nights[~test], cfg)
        p_full = m_full.predict_proba(X[test])[:, 1]
        p_base = m_base.predict_proba(X0[test])[:, 1]
        ll_f, br_f = _weighted_metrics(p_full, booked[test], nights[test])
        ll_b, br_b = _weighted_metrics(p_base, booked[test], nights[test])
        diagnostics.update(
            holdout_logloss_with_price=ll_f, holdout_logloss_without_price=ll_b,
            holdout_brier_with_price=br_f, holdout_brier_without_price=br_b,
        )
        dec = pd.DataFrame({"p": p_full, "b": booked[test], "n": nights[test]})
        dec["decile"] = pd.qcut(dec["p"].rank(method="first"), 10, labels=False) + 1
        calibration = dec.assign(pn=dec["p"] * dec["n"]).groupby("decile").agg(
            nights=("n", "sum"), predicted=("pn", "sum"), booked=("b", "sum"))
        calibration["predicted_rate"] = calibration["predicted"] / calibration["nights"]
        calibration["observed_rate"] = calibration["booked"] / calibration["nights"]
        calibration = calibration[["nights", "predicted_rate", "observed_rate"]].reset_index()
        _log(cfg, "holdout evaluation done", t1)

    # ---- final fit on all cells
    t1 = time.time()
    model = fit_logit(X, booked, nights, cfg)
    beta = model.coef_[0] * scales                     # back to the natural (prior-scaled) units
    _log(cfg, "final model fit", t1)

    a, _ = layout["slope_global"]
    slope_global = beta[a]
    ga, gb = layout["slope_group"]
    ca, cb = layout["slope_cohort"]
    group_dev = pd.Series(beta[ga:gb], index=cells["vehicle_type_group"].cat.categories)
    cohort_dev = pd.Series(beta[ca:cb], index=cells["cohort"].cat.categories)
    diagnostics["fixed_effects_slope_overall"] = float(slope_global)

    # ---- curves on each cohort's recent window
    eta = model.decision_function(X)
    recent_cut = cells["week"].max() - pd.Timedelta(weeks=cfg["curve_window_weeks"])
    win = cells["week"] > recent_cut
    cw = cells.loc[win, ["cohort", "vehicle_type_group", "nights", "booked", "log_price", "own_dev"]].copy()
    cw["eta"] = eta[win.to_numpy()]
    coh_g = cw["vehicle_type_group"].astype(str)
    cw["slope"] = slope_global + coh_g.map(group_dev).to_numpy() + cw["cohort"].astype(str).map(cohort_dev).to_numpy()
    price = np.exp(cw["log_price"].to_numpy())
    n_w = cw["nights"].to_numpy(float)

    curve_rows = []
    for k in cfg["price_grid"]:
        p = _sigmoid(cw["eta"].to_numpy() + cw["slope"].to_numpy() * np.log(k))
        tmp = pd.DataFrame({"cohort": cw["cohort"].to_numpy(), "n": n_w,
                            "bk": n_w * p, "rev": n_w * p * price * k})
        g = tmp.groupby("cohort", observed=True)[["n", "bk", "rev"]].sum()
        curve_rows.append(pd.DataFrame({
            "cohort": g.index.astype(str), "price_multiplier": k,
            "occupancy": g["bk"] / g["n"], "revpan": g["rev"] / g["n"]}))
    curves = pd.concat(curve_rows, ignore_index=True)

    # ---- per-cohort summary
    p_now = _sigmoid(cw["eta"].to_numpy())
    cw["p_now"] = p_now
    cw["np"] = n_w * p_now
    cw["np1p_s"] = n_w * p_now * (1 - p_now) * cw["slope"].to_numpy()
    meta = cells.groupby("cohort", observed=True).agg(
        vehicle_type_group=("vehicle_type_group", "first"),
        listings=("listing_id", "nunique"),
        nights_all=("nights", "sum"),
        booked_all=("booked", "sum"))
    chg = cells[cells["listing_changes_price"]].groupby("cohort", observed=True)["listing_id"].nunique()
    meta["listings_changing_price"] = chg.reindex(meta.index).fillna(0).astype(int)
    agg = cw.groupby("cohort", observed=True).agg(
        nights_recent=("nights", "sum"), booked_recent=("booked", "sum"),
        slope=("slope", "first"), np_sum=("np", "sum"), np1p_s=("np1p_s", "sum"),
        dev_p05=("own_dev", lambda s: s.quantile(0.05)),
        dev_p95=("own_dev", lambda s: s.quantile(0.95)))
    el = meta.join(agg, how="inner")
    el.index = el.index.astype(str)
    el["vehicle_type_group"] = el["vehicle_type_group"].astype(str)
    el["occupancy_recent"] = el["booked_recent"] / el["nights_recent"]
    # elasticity of expected booked nights w.r.t. price, at current prices
    el["elasticity_now"] = el["np1p_s"] / el["np_sum"]
    el["slope_group_level"] = slope_global + el["vehicle_type_group"].map(group_dev)

    best = curves.loc[curves.groupby("cohort")["revpan"].idxmax()].set_index("cohort")
    base = curves[np.isclose(curves["price_multiplier"], 1.0)].set_index("cohort")
    el["optimal_multiplier"] = best["price_multiplier"]
    el["revpan_now"] = base["revpan"]
    el["revpan_at_optimal"] = best["revpan"]
    el["revpan_lift_pct"] = 100 * (el["revpan_at_optimal"] / el["revpan_now"] - 1)
    el["occupancy_at_optimal"] = best["occupancy"]
    el["observed_range_low"] = np.exp(el["dev_p05"])
    el["observed_range_high"] = np.exp(el["dev_p95"])

    grid = cfg["price_grid"]
    el["flag_nonnegative_slope"] = el["slope"] >= 0
    el["flag_thin_data"] = ((el["listings_changing_price"] < cfg["min_listings_changing"]) |
                            (el["nights_all"] < cfg["min_nights"]))
    el["flag_optimum_at_grid_edge"] = el["optimal_multiplier"].isin([grid.min(), grid.max()])
    el["flag_optimum_outside_range"] = ~el["optimal_multiplier"].between(
        el["observed_range_low"] - 1e-9, el["observed_range_high"] + 1e-9)
    labels = {"flag_nonnegative_slope": "non-negative slope",
              "flag_thin_data": "thin data (leans on group slope)",
              "flag_optimum_at_grid_edge": "optimum at grid edge",
              "flag_optimum_outside_range": "optimum outside observed price changes"}
    el["flags"] = [
        "; ".join(lbl for col, lbl in labels.items() if row[col])
        for row in el[list(labels)].to_dict("records")]

    # Slope the recommender should use. A thin or wrong-signed cohort falls back to its group.
    use_group = el["flag_thin_data"] | el["flag_nonnegative_slope"]
    el["slope_used"] = np.where(use_group, el["slope_group_level"], el["slope"])
    el["slope_source"] = np.where(use_group, "group", "cohort")
    el["slope_overall"] = float(slope_global)

    el = el.reset_index().rename(columns={"index": "cohort"})
    split = el["cohort"].str.split(" | ", n=1, regex=False)
    el["metro"] = split.str[0]
    el["vehicle_type"] = split.str[1]
    el["elasticity_version"] = info["as_of"]
    el["run_at"] = pd.Timestamp.now().floor("s")
    el["booking_rate_overall"] = diagnostics["booking_rate"]
    el["share_listings_changing_price"] = diagnostics["share_listings_changing_price"]
    el["naive_slope_overall"] = diagnostics["naive_slope_no_fixed_effects"]

    el = el[[
        "elasticity_version", "run_at", "cohort", "metro", "vehicle_type", "vehicle_type_group",
        "listings", "listings_changing_price", "nights_all", "nights_recent", "occupancy_recent",
        "slope", "slope_group_level", "slope_overall", "slope_used", "slope_source",
        "elasticity_now", "optimal_multiplier", "occupancy_at_optimal", "revpan_now",
        "revpan_at_optimal", "revpan_lift_pct", "observed_range_low", "observed_range_high",
        "flag_thin_data", "flag_nonnegative_slope", "flag_optimum_at_grid_edge",
        "flag_optimum_outside_range", "flags",
        "booking_rate_overall", "share_listings_changing_price", "naive_slope_overall",
    ]].sort_values("nights_all", ascending=False).reset_index(drop=True)
    el = _plain_dtypes(el)

    _log(cfg, "done", t0)
    if cfg["verbose"]:
        _print_summary(diagnostics, el, calibration)

    # Small, plain dict only: pandas copies attrs on most operations.
    el.attrs["diagnostics"] = {k: v for k, v in diagnostics.items() if k != "controls"}
    if not return_details:
        return el
    curves.insert(0, "elasticity_version", info["as_of"])
    details = dict(curves=_plain_dtypes(curves), calibration=calibration,
                   diagnostics=diagnostics, model=model, layout=layout)
    return el, details


def _plain_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Plain numpy dtypes and str columns, so the frame queries and writes to SQL cleanly."""
    out = df.copy()
    for c in out.columns:
        dt = out[c].dtype
        if isinstance(dt, pd.CategoricalDtype) or dt == object or pd.api.types.is_string_dtype(dt):
            out[c] = out[c].astype(str).astype(object)
        elif pd.api.types.is_bool_dtype(dt):
            out[c] = out[c].astype(bool)
        elif pd.api.types.is_integer_dtype(dt):
            out[c] = out[c].astype("int64")
        elif pd.api.types.is_float_dtype(dt):
            out[c] = out[c].astype("float64")
    return out


# --------------------------------------------------------------------------- persistence
def save_elasticities(results: pd.DataFrame, path: str, smoothing: float = 0.7,
                      max_slope_change: float = 0.5) -> pd.DataFrame:
    """Append this run to the elasticity table that recommend() reads.

    `slope_applied` is what the recommender uses. It blends this run with the previous
    version (smoothing = weight on the new run) and caps the month-over-month change,
    so one noisy month can't whipsaw prices. The raw `slope_used` is kept alongside.

    path: .csv (default choice, no extra installs) or .parquet (needs pyarrow).
    Writing the same elasticity_version twice replaces it.
    """
    import os
    new = results.copy()
    version = new["elasticity_version"].iloc[0]
    old = _read_table(path) if os.path.exists(path) else pd.DataFrame()
    if len(old):
        old = old[old["elasticity_version"].astype(str) != str(version)]

    if len(old):
        last_v = old["elasticity_version"].astype(str).max()
        prev = old[old["elasticity_version"].astype(str) == last_v].set_index("cohort")["slope_applied"]
        p = new["cohort"].map(prev)
        blended = smoothing * new["slope_used"] + (1 - smoothing) * p
        capped = p + (blended - p).clip(-max_slope_change, max_slope_change)
        new["slope_applied"] = capped.fillna(new["slope_used"])   # new cohorts: raw value
        new["slope_previous"] = p
        new["previous_version"] = last_v
    else:
        new["slope_applied"] = new["slope_used"]
        new["slope_previous"] = np.nan
        new["previous_version"] = ""

    out = pd.concat([old, new], ignore_index=True) if len(old) else new
    if path.endswith(".parquet"):
        out.to_parquet(path, index=False)
    else:
        out.to_csv(path, index=False)
    n_cap = int(((new["slope_applied"] - new["slope_previous"]).abs() >= max_slope_change - 1e-9).sum())
    print(f"[price_response] saved version {version} ({len(new)} cohorts) to {path}"
          + (f"; {n_cap} cohort slopes hit the month-over-month cap" if n_cap else ""))
    return new


def _read_table(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={"elasticity_version": str, "previous_version": str},
                       keep_default_na=True)


def _print_summary(d, el, cal):
    print("\n=== Price response summary ===")
    print(f"Data: {d['date_min']} to {d['date_max']} (as of {d['as_of']}), "
          f"{d['listings']:,} listings, {d['cohorts']:,} cohorts, booking rate {d['booking_rate']:.1%}")
    print(f"Listings that ever changed base price: {d['share_listings_changing_price']:.0%} "
          f"({d['share_nights_from_price_changers']:.0%} of available nights)")
    print(f"Overall slope: naive (no fixed effects) {d['naive_slope_no_fixed_effects']:+.2f} -> "
          f"with fixed effects {d['fixed_effects_slope_overall']:+.2f}")
    if "holdout_logloss_with_price" in d:
        print(f"Holdout log loss: {d['holdout_logloss_without_price']:.4f} without price -> "
              f"{d['holdout_logloss_with_price']:.4f} with price")
    flagged = (el["flags"] != "").sum()
    print(f"Cohorts with flags: {flagged} of {len(el)}")
    cols = ["cohort", "nights_all", "occupancy_recent", "slope", "elasticity_now",
            "optimal_multiplier", "revpan_lift_pct", "flags"]
    with pd.option_context("display.width", 200, "display.max_colwidth", 45):
        print("\nLargest cohorts:")
        print(el[cols].head(10).to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    # Script use: load `history` however you normally do, e.g.
    #   history = pd.read_parquet("history.parquet")
    # then run this file, or import run_price_response from a notebook.
    try:
        history  # noqa: F821
    except NameError:
        raise SystemExit("Define a DataFrame named `history` first, then call run_price_response(history).")
    results = run_price_response(history)  # noqa: F821
