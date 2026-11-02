"""Customer master table with derived segments and RFM as of the end of the data."""
import numpy as np
import pandas as pd

# Store survey codes, as decoded in AdventureWorksDW (DimReseller.BusinessType)
BUSINESS_TYPES = {"BM": "Value Added Reseller", "BS": "Specialty Bike Shop", "OS": "Warehouse"}


def _rfm_segment(recency_days, orders):
    return np.select(
        [(recency_days <= 180) & (orders >= 3),
         recency_days <= 180,
         recency_days <= 365],
        ["Loyal active", "Active (1-2 orders)", "Cooling (6-12 months)"],
        "Lapsed (>12 months)",
    )


def build_customers(fact, t):
    """One row per buying customer: type, segment, demographics, territory, RFM and CM."""
    end_by_channel = fact.groupby("Channel").OrderDate.max()
    f = fact.assign(channel_end=fact.Channel.map(end_by_channel))

    orders = f.groupby(["CustomerID", "SalesOrderID"]).agg(OrderDate=("OrderDate", "first"),
                                                           channel_end=("channel_end", "first")).reset_index()
    c = orders.groupby("CustomerID").agg(first_order=("OrderDate", "min"), last_order=("OrderDate", "max"),
                                         orders=("SalesOrderID", "nunique"), channel_end=("channel_end", "max"))
    money = f.groupby("CustomerID").agg(revenue=("revenue", "sum"), cm=("cm", "sum"), cm1=("cm1", "sum"),
                                        units=("OrderQty", "sum"), Channel=("Channel", "first"),
                                        CustomerType=("CustomerType", "first"),
                                        Territory=("Territory", "first"), TerritoryGroup=("TerritoryGroup", "first"),
                                        ShipCountry=("ShipCountry", "first"),
                                        distance_km=("distance_km", "median"), is_distant=("is_distant", "max"))
    c = c.join(money).reset_index()
    c["recency_days"] = (c.channel_end - c.last_order).dt.days
    c["tenure_days"] = (c.channel_end - c.first_order).dt.days
    c["cm_margin"] = c.cm / c.revenue
    c["RFMSegment"] = _rfm_segment(c.recency_days, c.orders)

    cust = t["customer"][["CustomerID", "PersonID", "StoreID"]]
    c = c.merge(cust, on="CustomerID", how="left")

    # --- Store attributes -------------------------------------------------------
    s = t["store_demographics"].copy()
    s["StoreBusinessType"] = s.BusinessType.map(BUSINESS_TYPES).fillna(s.BusinessType)
    s["StoreSize"] = pd.cut(s.NumberEmployees, [0, 10, 50, 1000], labels=["Small (<=10)", "Medium (11-50)",
                                                                           "Large (>50)"]).astype(str)
    c = c.merge(s[["StoreID", "StoreName", "StoreBusinessType", "StoreSize", "Specialty", "AnnualSales",
                   "NumberEmployees"]], on="StoreID", how="left")

    # --- Individual demographics -------------------------------------------------
    p = t["person_demographics"].copy()
    p["IncomeBand"] = p.YearlyIncome
    ref = fact.OrderDate.max()
    p["Age"] = ((ref - p.BirthDate).dt.days / 365.25).round()
    c = c.merge(p[["PersonID", "IncomeBand", "Occupation", "Education", "Gender", "MaritalStatus", "Age",
                   "TotalChildren", "NumberCarsOwned", "HomeOwnerFlag"]], on="PersonID", how="left")
    c.loc[c.CustomerType == "Store", ["IncomeBand", "Occupation", "Education", "Gender", "MaritalStatus"]] = np.nan

    # --- Combined segment label ----------------------------------------------------
    c["Segment"] = np.where(c.CustomerType == "Store", "Store: " + c.StoreBusinessType.fillna("Unknown"),
                            "Individual: " + c.IncomeBand.fillna("Unknown"))
    not_applicable = pd.Series(np.where(c.CustomerType == "Store", "n/a (store)", "n/a (individual)"), index=c.index)
    for col in ["StoreBusinessType", "StoreSize", "IncomeBand", "Occupation"]:
        c[col] = c[col].fillna(not_applicable)
    return c.drop(columns=["channel_end"])
