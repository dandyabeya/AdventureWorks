"""Contribution margin fact table at order-line level."""
import numpy as np
import pandas as pd

from .distance import add_distance


def _price_at_date(lines, history, value_col):
    """Look up the history value valid at each line's OrderDate.

    Uses the latest period with StartDate <= OrderDate, so lines sold after a
    product's last period ended (discontinued stock) get the last known value.
    Returns (value, within_period) indexed by SalesOrderDetailID.
    """
    left = lines[["SalesOrderDetailID", "ProductID", "OrderDate"]].sort_values("OrderDate")
    m = pd.merge_asof(left, history.sort_values("StartDate"), left_on="OrderDate", right_on="StartDate",
                      by="ProductID", direction="backward").set_index("SalesOrderDetailID")
    within = m.EndDate.isna() | (m.OrderDate <= m.EndDate)
    return m[value_col], within & m[value_col].notna()


def build_fact(t, cfg):
    """Return the order-line fact table with revenue, COGS, freight allocation and CM1/CM2.

    CM1 = revenue - COGS; CM2 = CM1 - freight allocated by line revenue share.
    Column ``cm`` is CM2 if cfg['cm']['freight_allocation'] else CM1, and is the
    margin used by every downstream step.
    """
    hdr = t["sales_order_header"]
    det = t["sales_order_detail"]
    prod = t["product"].merge(t["product_subcategory"].rename(columns={"Name": "Subcategory"}),
                              on="ProductSubcategoryID", how="left")
    prod = prod.merge(t["product_category"].rename(columns={"Name": "Category"}), on="ProductCategoryID", how="left")
    prod[["Subcategory", "Category"]] = prod[["Subcategory", "Category"]].fillna("Unknown")
    terr = t["sales_territory"].rename(columns={"Name": "Territory", "CountryRegionCode": "TerritoryCountry"})
    addr = t["address"].merge(t["state_province"][["StateProvinceID", "StateProvinceCode", "Name", "CountryRegionCode"]]
                              .rename(columns={"Name": "StateProvince", "CountryRegionCode": "ShipCountry"}),
                              on="StateProvinceID", how="left")

    f = det.merge(hdr[["SalesOrderID", "OrderDate", "OnlineOrderFlag", "CustomerID", "TerritoryID",
                       "ShipToAddressID", "SubTotal", "Freight"]], on="SalesOrderID")
    f = f.merge(prod[["ProductID", "Name", "ProductNumber", "Subcategory", "Category", "StandardCost", "ListPrice"]]
                .rename(columns={"Name": "Product", "StandardCost": "current_cost", "ListPrice": "current_list_price"}),
                on="ProductID", how="left")
    f = f.merge(t["customer"][["CustomerID", "CustomerType"]], on="CustomerID", how="left")
    f = f.merge(terr, on="TerritoryID", how="left")
    f = f.merge(addr[["AddressID", "City", "StateProvinceCode", "StateProvince", "ShipCountry", "Latitude", "Longitude"]],
                left_on="ShipToAddressID", right_on="AddressID", how="left").drop(columns="AddressID")
    f = f.merge(t["special_offer"][["SpecialOfferID", "Type"]].rename(columns={"Type": "OfferType"}),
                on="SpecialOfferID", how="left")

    # --- Unit cost and list price valid at order date -------------------------
    hist_cost, cost_within = _price_at_date(f, t["product_cost_history"], "StandardCost")
    hist_list, _ = _price_at_date(f, t["product_list_price_history"], "ListPrice")
    f["hist_cost"] = f.SalesOrderDetailID.map(hist_cost)
    f["list_price"] = f.SalesOrderDetailID.map(hist_list).fillna(f.current_list_price)
    if cfg["cm"]["cost_basis"] == "historical":
        f["unit_cost"] = f.hist_cost.fillna(f.current_cost)
    else:
        f["unit_cost"] = f.current_cost
    # False = sold after the last cost period ended (last known cost used) or no history at all
    f["cost_within_period"] = f.SalesOrderDetailID.map(cost_within).fillna(False).astype(bool)

    # --- Revenue, COGS, freight, CM --------------------------------------------
    f["is_online"] = f.OnlineOrderFlag.astype(bool)
    f["Channel"] = np.where(f.is_online, "Online", "Reseller")
    f["revenue"] = f.LineTotal
    f["gross_revenue"] = f.UnitPrice * f.OrderQty
    f["discount_amount"] = f.gross_revenue - f.revenue
    f["effective_price"] = f.UnitPrice * (1 - f.UnitPriceDiscount)
    f["price_index"] = f.effective_price / f.list_price
    f["cogs"] = f.unit_cost * f.OrderQty
    order_revenue = f.groupby("SalesOrderID").revenue.transform("sum")
    f["freight_alloc"] = f.Freight * f.revenue / order_revenue
    f["cm1"] = f.revenue - f.cogs
    f["cm2"] = f.cm1 - f.freight_alloc
    f["cm"] = f.cm2 if cfg["cm"]["freight_allocation"] else f.cm1
    f["cm_margin"] = f.cm / f.revenue

    # --- Time and distance -------------------------------------------------------
    f["OrderMonth"] = f.OrderDate.dt.to_period("M").dt.to_timestamp()
    f["Year"] = f.OrderDate.dt.year
    f["Quarter"] = f.OrderDate.dt.quarter
    f["Month"] = f.OrderDate.dt.month
    f = add_distance(f, cfg)

    return f.drop(columns=["SubTotal", "Freight", "OnlineOrderFlag", "hist_cost"])


def cm_summary(fact):
    """CM1 vs CM2 by channel, for reporting both freight variants."""
    g = fact.groupby("Channel")[["revenue", "cogs", "freight_alloc", "cm1", "cm2"]].sum()
    g.loc["Total"] = g.sum()
    g["cm1_margin"] = g.cm1 / g.revenue
    g["cm2_margin"] = g.cm2 / g.revenue
    return g.reset_index()
