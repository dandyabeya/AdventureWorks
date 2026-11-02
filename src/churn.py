"""Customer churn model: RFM + CM + product-mix features, logistic regression and LightGBM.

Churn at snapshot date S = the customer bought on or before S but places no
order in the following ``window_months``. Features only use data up to S.

Evaluation is out-of-time: train on the validation snapshot (features up to
S_val, label in the next window) and test on the main snapshot S. The final
model is refit on both snapshots and scores every customer as of the end of the
data, i.e. the probability of not buying in the next ``window_months``.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

NUMERIC = ["recency_days", "frequency", "monetary", "cm_total", "cm_margin", "tenure_days", "avg_order_value",
           "units", "n_products", "n_subcategories", "share_bikes", "share_components", "share_clothing",
           "share_accessories", "avg_price_index", "discount_share", "orders_last_90d", "orders_last_180d",
           "revenue_last_180d", "distance_km", "Age", "TotalChildren", "NumberCarsOwned", "HomeOwnerFlag",
           "AnnualSales", "NumberEmployees"]
CATEGORICAL = ["CustomerType", "Territory", "TerritoryGroup", "IncomeBand", "Education", "Occupation", "Gender",
               "MaritalStatus", "StoreBusinessType", "StoreSize", "Specialty"]
SKEWED = ["monetary", "cm_total", "avg_order_value", "units", "revenue_last_180d", "AnnualSales"]
STATIC = ["CustomerID", "CustomerType", "Territory", "TerritoryGroup", "distance_km", "IncomeBand", "Education",
          "Occupation", "Gender", "MaritalStatus", "Age", "TotalChildren", "NumberCarsOwned", "HomeOwnerFlag",
          "StoreBusinessType", "StoreSize", "Specialty", "AnnualSales", "NumberEmployees"]


def features_at(fact, customers, snapshot):
    """Customer features using only order lines up to ``snapshot``."""
    snapshot = pd.Timestamp(snapshot)
    f = fact[fact.OrderDate <= snapshot]
    g = f.groupby("CustomerID")
    x = pd.DataFrame({
        "first_order": g.OrderDate.min(), "last_order": g.OrderDate.max(),
        "frequency": g.SalesOrderID.nunique(), "monetary": g.revenue.sum(), "cm_total": g.cm.sum(),
        "units": g.OrderQty.sum(), "n_products": g.ProductID.nunique(), "n_subcategories": g.Subcategory.nunique(),
        "avg_price_index": g.price_index.mean(),
        "discount_share": g.discount_amount.sum() / g.gross_revenue.sum(),
    })
    x["recency_days"] = (snapshot - x.last_order).dt.days
    x["tenure_days"] = (snapshot - x.first_order).dt.days
    x["cm_margin"] = x.cm_total / x.monetary
    x["avg_order_value"] = x.monetary / x.frequency
    mix = f.pivot_table(index="CustomerID", columns="Category", values="revenue", aggfunc="sum", fill_value=0)
    mix = mix.div(mix.sum(axis=1), axis=0)
    for cat in ["Bikes", "Components", "Clothing", "Accessories"]:
        x[f"share_{cat.lower()}"] = mix[cat] if cat in mix else 0.0
    for days in (90, 180):
        recent = f[f.OrderDate > snapshot - pd.Timedelta(days=days)]
        x[f"orders_last_{days}d"] = recent.groupby("CustomerID").SalesOrderID.nunique()
    x["revenue_last_180d"] = f[f.OrderDate > snapshot - pd.Timedelta(days=180)].groupby("CustomerID").revenue.sum()
    x[["orders_last_90d", "orders_last_180d", "revenue_last_180d"]] = x[
        ["orders_last_90d", "orders_last_180d", "revenue_last_180d"]].fillna(0)
    x = x.reset_index().merge(customers[STATIC], on="CustomerID", how="left")
    x["snapshot"] = snapshot
    return x.drop(columns=["first_order", "last_order"])


def label_at(fact, snapshot, months):
    """1 = no order in (snapshot, snapshot + months], for customers active up to the snapshot."""
    snapshot = pd.Timestamp(snapshot)
    end = snapshot + pd.DateOffset(months=months)
    future = fact[(fact.OrderDate > snapshot) & (fact.OrderDate <= end)].CustomerID.unique()
    past = fact[fact.OrderDate <= snapshot].CustomerID.unique()
    return pd.Series(~np.isin(past, future), index=past, name="churn").astype(int)


def dataset(fact, customers, snapshot, months):
    x = features_at(fact, customers, snapshot)
    y = label_at(fact, snapshot, months)
    return x.merge(y, left_on="CustomerID", right_index=True)


def _preprocessor(scale):
    log_tf = FunctionTransformer(lambda a: np.sign(a) * np.log1p(np.abs(a)), feature_names_out="one-to-one")
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    plain = [c for c in NUMERIC if c not in SKEWED]
    return ColumnTransformer([
        ("skewed", Pipeline([("log", log_tf)] + num_steps), SKEWED),
        ("num", Pipeline(num_steps), plain),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), CATEGORICAL),
    ])


def make_models(random_state=42):
    return {
        "logistic_regression": Pipeline([("prep", _preprocessor(scale=True)),
                                         ("clf", LogisticRegression(max_iter=2000, C=1.0))]),
        "lightgbm": Pipeline([("prep", _preprocessor(scale=False)),
                              ("clf", lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                                                         min_child_samples=40, subsample=0.8, subsample_freq=1,
                                                         colsample_bytree=0.8, random_state=random_state,
                                                         verbose=-1))]),
    }


def metrics(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    top = np.argsort(-p)[: max(1, len(p) // 10)]
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p),
            "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred), "base_rate": float(np.mean(y)),
            "top_decile_churn_rate": float(np.mean(np.asarray(y)[top]))}


def train_and_evaluate(fact, customers, cfg):
    """Out-of-time evaluation, then final fit on both snapshots and scoring at data end.

    Returns dict with metrics, importances, the fitted final model and scored customers.
    """
    cc = cfg["churn"]
    months, rs = cc["window_months"], cc["random_state"]
    train = dataset(fact, customers, cc["validation_snapshot_date"], months)
    test = dataset(fact, customers, cc["snapshot_date"], months)
    X_cols = NUMERIC + CATEGORICAL

    rows, fitted = [], {}
    for name, model in make_models(rs).items():
        model.fit(train[X_cols], train.churn)
        p = model.predict_proba(test[X_cols])[:, 1]
        fitted[name] = model
        rows.append({"model": name, "segment": "All", **metrics(test.churn, p)})
        for ctype in ["Individual", "Store"]:
            m = (test.CustomerType == ctype).values
            if m.sum() and test.churn[m].nunique() == 2:
                rows.append({"model": name, "segment": ctype, **metrics(test.churn[m], p[m])})
    metric_table = pd.DataFrame(rows)

    # --- Feature importance ------------------------------------------------------
    lgbm = fitted["lightgbm"]
    perm = permutation_importance(lgbm, test[X_cols], test.churn, scoring="roc_auc", n_repeats=5,
                                  random_state=rs, n_jobs=1)
    importance = pd.DataFrame({"feature": X_cols, "permutation_importance_auc": perm.importances_mean,
                               "permutation_std": perm.importances_std}).sort_values(
        "permutation_importance_auc", ascending=False)
    lr = fitted["logistic_regression"]
    coefs = pd.DataFrame({"feature": lr.named_steps["prep"].get_feature_names_out(),
                          "coefficient": lr.named_steps["clf"].coef_[0]}).sort_values("coefficient", key=np.abs,
                                                                                      ascending=False)

    # --- Final model on both labelled snapshots, score at data end ------------------
    best = metric_table[metric_table.segment == "All"].sort_values("roc_auc", ascending=False).model.iloc[0]
    final = make_models(rs)[best]
    stacked = pd.concat([train, test], ignore_index=True)
    final.fit(stacked[X_cols], stacked.churn)
    # each channel is scored at its own last order date (reseller data ends earlier)
    current = pd.concat([features_at(fact[fact.Channel == ch], customers, end)
                         for ch, end in fact.groupby("Channel").OrderDate.max().items()], ignore_index=True)
    current["churn_prob"] = final.predict_proba(current[X_cols])[:, 1]

    return {"metrics": metric_table, "permutation_importance": importance, "lr_coefficients": coefs,
            "best_model": best, "final_model": final, "scored": current[["CustomerID", "churn_prob"]],
            "train_size": len(train), "test_size": len(test),
            "train_base_rate": float(train.churn.mean()), "test_base_rate": float(test.churn.mean())}
