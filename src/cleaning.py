"""Load raw tables, validate and clean them, and log every check.

Every check appends one row to the cleaning log (check, table, rows affected,
action), which is exported so the cleaning steps can be shown and defended.
"""
import numpy as np
import pandas as pd

DATE_COLS = {
    "sales_order_header": ["OrderDate", "DueDate", "ShipDate"],
    "product": ["SellStartDate", "SellEndDate", "DiscontinuedDate"],
    "product_cost_history": ["StartDate", "EndDate"],
    "product_list_price_history": ["StartDate", "EndDate"],
    "person_demographics": ["BirthDate"],
    "special_offer": ["StartDate", "EndDate"],
}
PRIMARY_KEYS = {
    "sales_order_header": ["SalesOrderID"],
    "sales_order_detail": ["SalesOrderDetailID"],
    "product": ["ProductID"],
    "product_subcategory": ["ProductSubcategoryID"],
    "product_category": ["ProductCategoryID"],
    "product_cost_history": ["ProductID", "StartDate"],
    "product_list_price_history": ["ProductID", "StartDate"],
    "customer": ["CustomerID"],
    "store_demographics": ["StoreID"],
    "person_demographics": ["PersonID"],
    "sales_territory": ["TerritoryID"],
    "address": ["AddressID"],
    "state_province": ["StateProvinceID"],
    "special_offer": ["SpecialOfferID"],
    "ship_method": ["ShipMethodID"],
}
# (child table, child column, parent table, parent column)
FOREIGN_KEYS = [
    ("sales_order_detail", "SalesOrderID", "sales_order_header", "SalesOrderID"),
    ("sales_order_detail", "ProductID", "product", "ProductID"),
    ("sales_order_detail", "SpecialOfferID", "special_offer", "SpecialOfferID"),
    ("sales_order_header", "CustomerID", "customer", "CustomerID"),
    ("sales_order_header", "TerritoryID", "sales_territory", "TerritoryID"),
    ("sales_order_header", "ShipToAddressID", "address", "AddressID"),
    ("address", "StateProvinceID", "state_province", "StateProvinceID"),
]


def load_raw(raw_dir):
    tables = {}
    for name in PRIMARY_KEYS:
        tables[name] = pd.read_csv(raw_dir / f"{name}.csv", parse_dates=DATE_COLS.get(name, []))
    return tables


class CleaningLog:
    def __init__(self):
        self.rows = []

    def add(self, check, table, rows_affected, action):
        self.rows.append({"check": check, "table": table, "rows_affected": int(rows_affected), "action": action})

    def frame(self):
        return pd.DataFrame(self.rows)


def clean(tables):
    """Return (cleaned tables, cleaning log DataFrame). Input dict is not modified."""
    t = {k: v.copy() for k, v in tables.items()}
    log = CleaningLog()

    # --- Generic checks: strings, duplicates, primary keys --------------------
    for name, df in t.items():
        for col in df.select_dtypes("object"):
            df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)
        dups = df.duplicated().sum()
        if dups:
            df.drop_duplicates(inplace=True)
        log.add("exact duplicate rows", name, dups, "dropped" if dups else "none found")
        pk_dups = df.duplicated(PRIMARY_KEYS[name]).sum()
        if pk_dups:
            df.drop_duplicates(PRIMARY_KEYS[name], keep="first", inplace=True)
        log.add(f"duplicate primary key {PRIMARY_KEYS[name]}", name, pk_dups, "kept first" if pk_dups else "none found")

    # --- Referential integrity --------------------------------------------------
    for child, ccol, parent, pcol in FOREIGN_KEYS:
        orphan = ~t[child][ccol].isin(t[parent][pcol])
        if orphan.any():
            t[child] = t[child][~orphan]
        log.add(f"orphan {ccol} (not in {parent})", child, orphan.sum(), "dropped" if orphan.any() else "none found")

    hdr, det = t["sales_order_header"], t["sales_order_detail"]

    # --- Order lines: quantities, prices, discounts, LineTotal -----------------
    bad = (det.OrderQty <= 0) | (det.UnitPrice <= 0) | ~det.UnitPriceDiscount.between(0, 1, inclusive="left")
    t["sales_order_detail"] = det = det[~bad]
    log.add("OrderQty<=0, UnitPrice<=0 or discount outside [0,1)", "sales_order_detail", bad.sum(),
            "dropped" if bad.any() else "none found")

    recomputed = det.UnitPrice * det.OrderQty * (1 - det.UnitPriceDiscount)
    off = (det.LineTotal - recomputed).abs() > 0.01
    log.add("LineTotal differs from UnitPrice*OrderQty*(1-discount) by > 0.01", "sales_order_detail", off.sum(),
            "LineTotal kept (source of truth, rounding only)")

    lines_per_order = det.groupby("SalesOrderID").LineTotal.sum()
    sub_diff = (hdr.set_index("SalesOrderID").SubTotal - lines_per_order).abs() > 0.05
    log.add("SubTotal differs from sum of LineTotal by > 0.05", "sales_order_header", sub_diff.sum(),
            "kept; freight is allocated on the sum of LineTotal")

    no_lines = ~hdr.SalesOrderID.isin(det.SalesOrderID)
    t["sales_order_header"] = hdr = hdr[~no_lines]
    log.add("orders without order lines", "sales_order_header", no_lines.sum(), "dropped" if no_lines.any() else "none found")

    not_shipped = hdr.Status != 5
    log.add("orders not in status 5 (shipped)", "sales_order_header", not_shipped.sum(),
            "kept (all are shipped)" if not not_shipped.any() else "kept, flagged")

    neg_freight = hdr.Freight < 0
    log.add("negative freight", "sales_order_header", neg_freight.sum(), "none found" if not neg_freight.any() else "kept, flagged")

    # --- Coverage: channel end dates ---------------------------------------------
    last = hdr.groupby("OnlineOrderFlag").OrderDate.max()
    for flag, d in last.items():
        log.add(f"last order date {'online' if flag else 'reseller'} = {d.date()}", "sales_order_header", 0,
                "info: baselines are measured per channel")

    # --- Cost / list price history: end date is inclusive -----------------------
    for name in ["product_cost_history", "product_list_price_history"]:
        h = t[name]
        gaps = (h.sort_values("StartDate").groupby("ProductID").apply(
            lambda g: (g.StartDate.iloc[1:].values - g.EndDate.iloc[:-1].values > np.timedelta64(1, "D")).sum(),
            include_groups=False)).sum()
        log.add("gaps between history periods > 1 day", name, gaps, "none found" if not gaps else "fallback to current value")
        log.add("EndDate is the last valid day (inclusive)", name, 0, "joins use StartDate <= OrderDate <= EndDate")

    # --- Address coordinates at (0, 0) -------------------------------------------
    addr = t["address"]
    zero = (addr.Latitude.abs() < 1e-6) & (addr.Longitude.abs() < 1e-6)
    if zero.any():
        valid = addr[~zero]
        city = valid.groupby(["City", "StateProvinceID"])[["Latitude", "Longitude"]].median()
        state = valid.groupby("StateProvinceID")[["Latitude", "Longitude"]].median()
        for idx in addr.index[zero]:
            key = (addr.at[idx, "City"], addr.at[idx, "StateProvinceID"])
            src = city.loc[key] if key in city.index else state.loc[addr.at[idx, "StateProvinceID"]]
            addr.loc[idx, ["Latitude", "Longitude"]] = src.values
    log.add("address coordinates at (0, 0)", "address", zero.sum(),
            "imputed with median of same city (else same state)" if zero.any() else "none found")
    missing_geo = addr[["Latitude", "Longitude"]].isna().any(axis=1)
    log.add("address coordinates missing", "address", missing_geo.sum(), "none found" if not missing_geo.any() else "left missing")

    # --- Products sold without category -----------------------------------------
    sold = t["product"][t["product"].ProductID.isin(det.ProductID)]
    no_cat = sold.ProductSubcategoryID.isna()
    log.add("sold products without subcategory", "product", no_cat.sum(),
            "none found" if not no_cat.any() else "category set to 'Unknown'")

    # --- Customers: type --------------------------------------------------------
    cust = t["customer"]
    cust["CustomerType"] = np.where(cust.StoreID.notna(), "Store", np.where(cust.PersonID.notna(), "Individual", "Unknown"))
    log.add("customers with neither StoreID nor PersonID", "customer", (cust.CustomerType == "Unknown").sum(),
            "labelled 'Unknown'")

    # --- Demographics: normalise categorical text --------------------------------
    pdem = t["person_demographics"]
    log.add("person demographics with missing YearlyIncome", "person_demographics", pdem.YearlyIncome.isna().sum(),
            "left missing")

    return t, log.frame()


def save_processed(tables, log, processed_dir):
    for name, df in tables.items():
        df.to_parquet(processed_dir / f"{name}.parquet", index=False)
    log.to_csv(processed_dir / "cleaning_log.csv", index=False)


def load_processed(processed_dir):
    return {name: pd.read_parquet(processed_dir / f"{name}.parquet") for name in PRIMARY_KEYS}
