"""Tidy CSV exports for Power BI (schemas documented in SCHEMA.md)."""
import numpy as np
import pandas as pd

FACT_COLUMNS = [
    "SalesOrderID", "SalesOrderDetailID", "OrderDate", "OrderMonth", "Year", "Quarter", "Month",
    "CustomerID", "CustomerType", "Channel", "ProductID", "Product", "Subcategory", "Category",
    "SpecialOfferID", "OfferType", "OrderQty", "UnitPrice", "UnitPriceDiscount", "list_price", "effective_price",
    "price_index", "gross_revenue", "discount_amount", "revenue", "unit_cost", "cogs", "freight_alloc",
    "cm1", "cm2", "cm", "cm_margin", "cost_within_period", "TerritoryID", "Territory", "TerritoryGroup",
    "ShipCountry", "StateProvince", "City", "Latitude", "Longitude", "distance_km", "distance_band", "is_distant",
]

# dimension -> (key column, name column, parent column) in the pareto.run_all tables
PARETO_KEYS = {
    "customer": ("CustomerID", "CustomerID", None),
    "product": ("ProductID", "Product", "Category"),
    "subcategory": ("Subcategory", "Subcategory", "Category"),
    "territory": ("Territory", "Territory", "TerritoryGroup"),
    "channel_category": ("Category", "Category", "Channel"),
}
PARETO_COLUMNS = ["revenue", "cogs", "cm", "cm_margin", "lines", "rank", "entity_share", "cm_share", "cum_cm_share",
                  "cum_positive_cm_share", "abc_class"]


def pareto_long(results):
    """Stack all ABC tables into one long table keyed by dimension."""
    frames = []
    for dim, g in results.items():
        key, name, parent = PARETO_KEYS.get(dim, (g.columns[0], g.columns[0], None))
        out = pd.DataFrame({"dimension": dim, "entity_key": g[key].astype(str), "entity_name": g[name].astype(str),
                            "parent": g[parent].astype(str) if parent else ""})
        frames.append(pd.concat([out, g[PARETO_COLUMNS]], axis=1))
    return pd.concat(frames, ignore_index=True)


def risk_band(p):
    return pd.cut(p, [-0.01, 0.3, 0.6, 0.8, 1.0], labels=["Low (<30%)", "Medium (30-60%)", "High (60-80%)",
                                                          "Very high (>80%)"]).astype(str)


def write_all(out_dir, fact, customers, pareto_results, product_abc, low_cm_sets, churn_result, elasticity_table,
              scenario_segments, scenario_total, scenario_optimum, extras):
    """Write every CSV and return {file name: row count}."""
    files = {}

    def save(name, df):
        df.to_csv(out_dir / name, index=False, date_format="%Y-%m-%d")
        files[name] = len(df)

    save("fact_order_lines.csv", fact[FACT_COLUMNS])

    prod = product_abc.rename(columns={"abc_class": "abc_class_cm"})
    for rule, ids in low_cm_sets.items():
        prod[f"low_cm_{rule}"] = prod.ProductID.isin(ids)
    save("dim_product.csv", prod)

    cust_abc = pareto_results["customer"][["CustomerID", "rank", "abc_class"]].rename(
        columns={"rank": "cm_rank", "abc_class": "abc_class_cm"})
    cust = customers.merge(cust_abc, on="CustomerID", how="left").merge(churn_result["scored"], on="CustomerID",
                                                                        how="left")
    cust["churn_risk_band"] = risk_band(cust.churn_prob)
    save("dim_customer.csv", cust)

    save("pareto_abc.csv", pareto_long(pareto_results))
    save("pareto_concentration.csv", extras["concentration"])
    save("abc_class_summary.csv", extras["class_summary"])
    save("cm_summary.csv", extras["cm_summary"])
    save("distance_comparison.csv", extras["distance_comparison"])
    save("elasticity.csv", elasticity_table)
    save("elasticity_price_variation.csv", extras["within_variation"])
    save("churn_metrics.csv", churn_result["metrics"])
    save("churn_feature_importance.csv", churn_result["permutation_importance"])
    save("churn_lr_coefficients.csv", churn_result["lr_coefficients"])
    save("scenario_results.csv", scenario_segments)
    save("scenario_summary.csv", scenario_total)
    save("scenario_optimum.csv", scenario_optimum)
    save("cleaning_log.csv", extras["cleaning_log"])
    return files
