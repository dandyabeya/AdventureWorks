"""Price elasticity estimation from observed price variation, with assumed fallbacks.

Two estimators on a product x territory x channel x month panel:

1. Log-log fixed-effects OLS (per channel):
   log(units) ~ log(price) + log(price):Category + log(price):TerritoryGroup
                + product x territory FE + month FE
   so the elasticity is identified from price changes *within* a product in a
   territory over time (list price changes, promotions), not from cheap vs.
   expensive products.
2. LightGBM on log(units) with the price index (effective / list price) plus
   controls; elasticity = response of predicted units to a +/-5% price change.

Estimates that are not significant or fall outside the plausible range are
replaced by the clearly labelled assumed "mid" elasticity.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

FEATURES = ["log_list_price", "price_index", "Category", "Subcategory", "TerritoryGroup", "Territory",
            "Channel", "Month", "Year"]
CATEGORICAL = ["Category", "Subcategory", "TerritoryGroup", "Territory", "Channel"]


def build_panel(fact, cfg):
    """Aggregate order lines to product x territory x channel x month cells."""
    f = fact
    if cfg["elasticity"]["exclude_volume_discounts"]:
        f = f[f.OfferType != "Volume Discount"]
    p = f.groupby(["ProductID", "Category", "Subcategory", "Territory", "TerritoryGroup", "Channel", "OrderMonth"],
                  observed=True).agg(units=("OrderQty", "sum"), revenue=("revenue", "sum"),
                                     list_revenue=("list_price", lambda s: (s * f.loc[s.index, "OrderQty"]).sum()),
                                     orders=("SalesOrderID", "nunique")).reset_index()
    p["price"] = p.revenue / p.units
    p["list_price"] = p.list_revenue / p.units
    p["price_index"] = p.price / p.list_price
    p["log_units"] = np.log(p.units)
    p["log_price"] = np.log(p.price)
    p["log_list_price"] = np.log(p.list_price)
    p["Month"] = p.OrderMonth.dt.month
    p["Year"] = p.OrderMonth.dt.year
    p["fe_unit"] = p.ProductID.astype(str) + "_" + p.Territory
    return p


def within_price_variation(panel):
    """How much does log price move within a product x territory unit? (identification check)."""
    demeaned = panel.log_price - panel.groupby(["Channel", "fe_unit"]).log_price.transform("mean")
    return panel.assign(dm=demeaned).groupby("Channel").agg(
        cells=("dm", "size"), sd_within_log_price=("dm", "std"),
        share_cells_with_price_change=("dm", lambda s: (s.abs() > 0.01).mean())).reset_index()


def fit_ols(panel, channel):
    """Fixed-effects log-log regression for one channel. Returns (results, data used)."""
    d = panel[panel.Channel == channel].copy()
    # keep units with price variation; others are absorbed by the FE and only add dummies
    d = d[d.groupby("fe_unit").log_price.transform("std").fillna(0) > 1e-6]
    formula = ("log_units ~ log_price + log_price:C(Category, Treatment('Bikes')) "
               "+ log_price:C(TerritoryGroup, Treatment('North America')) + C(fe_unit) + C(OrderMonth)")
    res = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d.ProductID})
    return res, d


def ols_cell_elasticities(res, d, channel):
    """Elasticity (with SE and p-value) for each category x territory group cell."""
    names = res.params.index
    rows = []
    for cat in sorted(d.Category.unique()):
        for grp in sorted(d.TerritoryGroup.unique()):
            terms = ["log_price"]
            cat_term = f"log_price:C(Category, Treatment('Bikes'))[T.{cat}]"
            grp_term = f"log_price:C(TerritoryGroup, Treatment('North America'))[T.{grp}]"
            terms += [t for t in (cat_term, grp_term) if t in names]
            weights = np.array([1.0 if n in terms else 0.0 for n in names])
            test = res.t_test(weights)
            n = int(((d.Category == cat) & (d.TerritoryGroup == grp)).sum())
            rows.append({"Channel": channel, "Category": cat, "TerritoryGroup": grp,
                         "elasticity_ols": float(test.effect[0]), "se_ols": float(test.sd[0][0]),
                         "p_ols": float(test.pvalue), "n_cells": n})
    return pd.DataFrame(rows)


def fit_lgbm(panel, random_state=42):
    X = panel[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=30,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              random_state=random_state, verbose=-1)
    model.fit(X, panel.log_units)
    return model


def lgbm_cell_elasticities(model, panel, delta=0.05):
    """Arc elasticity from predicted log units at price_index x (1 +/- delta), per cell."""
    X = panel[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype(panel[c].astype("category").dtype)
    up, down = X.copy(), X.copy()
    up["price_index"] *= 1 + delta
    down["price_index"] *= 1 - delta
    e = (model.predict(up) - model.predict(down)) / (np.log(1 + delta) - np.log(1 - delta))
    return panel.assign(e=e).groupby(["Channel", "Category", "TerritoryGroup"]).e.mean().rename(
        "elasticity_lgbm").reset_index()


def estimate(fact, cfg):
    """Run both estimators and build the elasticity table used by the scenario engine.

    Returns (table, diagnostics dict).
    """
    ec = cfg["elasticity"]
    lo, hi = ec["plausible_range"]
    panel = build_panel(fact, cfg)

    ols_tables, ols_summary = [], {}
    for channel in sorted(panel.Channel.unique()):
        res, d = fit_ols(panel, channel)
        ols_tables.append(ols_cell_elasticities(res, d, channel))
        ols_summary[channel] = {"n": int(res.nobs), "r2_within_model": float(res.rsquared),
                                "base_log_price_coef": float(res.params["log_price"]),
                                "base_log_price_p": float(res.pvalues["log_price"])}
    table = pd.concat(ols_tables, ignore_index=True)

    model = fit_lgbm(panel, cfg["churn"]["random_state"])
    table = table.merge(lgbm_cell_elasticities(model, panel), on=["Channel", "Category", "TerritoryGroup"], how="left")

    table["plausible"] = (table.elasticity_ols.between(lo, hi) & (table.p_ols < 0.05)
                          & (table.n_cells >= ec["min_obs"]))
    table["elasticity_estimated"] = np.where(table.plausible, table.elasticity_ols, ec["assumed"]["mid"])
    table["estimated_source"] = np.where(table.plausible, "estimated (OLS)", "assumed mid (estimate not credible)")
    for name, value in ec["assumed"].items():
        table[f"elasticity_{name}"] = value

    diagnostics = {"panel_cells": len(panel), "within_variation": within_price_variation(panel),
                   "ols": ols_summary, "lgbm_feature_importance": pd.Series(
                       model.booster_.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)}
    return table, diagnostics


def lookup(table, scenario):
    """Elasticity per (Channel, Category, TerritoryGroup) for an elasticity scenario name."""
    col = "elasticity_estimated" if scenario == "estimated" else f"elasticity_{scenario}"
    return table.set_index(["Channel", "Category", "TerritoryGroup"])[col]
