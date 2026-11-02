"""Great-circle distance from HQ and the 'distant region' flag."""
import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def add_distance(df, cfg, lat="Latitude", lon="Longitude"):
    """Add distance_km, distance_band and is_distant columns (ship-to address based)."""
    d = cfg["distance"]
    df = df.copy()
    df["distance_km"] = haversine_km(d["hq_lat"], d["hq_lon"], df[lat], df[lon]).round(1)
    edges = d["bands_km"]
    labels = [f"{edges[i]:,}-{edges[i + 1]:,} km" for i in range(len(edges) - 1)]
    df["distance_band"] = pd.cut(df.distance_km, edges, labels=labels, right=False).astype(str)
    df["is_distant"] = df.distance_km > d["distant_km"]
    return df


def compare_definitions(fact, cfg):
    """Compare the three candidate definitions of 'distant' on the order-line fact table.

    Returns one row per territory with the share of revenue each option flags as
    distant, plus the freight-to-revenue ratio (to test freight as a proxy).
    """
    f = fact.assign(
        opt_territory_group=fact.TerritoryGroup != "North America",
        opt_great_circle=fact.is_distant,
    )
    orders = f.groupby("SalesOrderID").agg(freight=("freight_alloc", "sum"), revenue=("revenue", "sum"),
                                           units=("OrderQty", "sum"))
    freight_per_unit = orders.freight / orders.units
    threshold = freight_per_unit.quantile(0.75)
    f["opt_freight_per_unit"] = f.SalesOrderID.map(freight_per_unit > threshold)

    out = f.groupby(["TerritoryGroup", "Territory"]).apply(lambda g: pd.Series({
        "median_distance_km": g.distance_km.median(),
        "revenue": g.revenue.sum(),
        "freight_pct_of_revenue": g.freight_alloc.sum() / g.revenue.sum(),
        "share_distant_territory_group": np.average(g.opt_territory_group, weights=g.revenue),
        "share_distant_great_circle": np.average(g.opt_great_circle, weights=g.revenue),
        "share_distant_freight_per_unit": np.average(g.opt_freight_per_unit, weights=g.revenue),
    }), include_groups=False).reset_index().sort_values("median_distance_km")

    order_level = f.groupby("SalesOrderID").agg(distance_km=("distance_km", "first"), online=("is_online", "first"))
    order_level["freight_ratio"] = orders.freight / orders.revenue
    corr = {
        "corr_freight_ratio_vs_distance_online": order_level[order_level.online].freight_ratio.corr(
            order_level[order_level.online].distance_km),
        "corr_freight_ratio_vs_distance_reseller": order_level[~order_level.online].freight_ratio.corr(
            order_level[~order_level.online].distance_km),
    }
    return out, corr
