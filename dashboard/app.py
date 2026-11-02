"""AdventureWorks markup simulation dashboard (Streamlit).

Reads the artefacts written by ``python -m src.pipeline`` and runs the scenario
engine live, so any markup level / elasticity / churn assumption can be tried.

    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import scenarios  # noqa: E402
from src.config import load_config  # noqa: E402

# Categorical slots in fixed order (validated palette), ABC as an ordinal blue ramp, losses = red
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
LOSS = "#e34948"

# --- Plain-language labels ---------------------------------------------------------
TYPE_LABELS = {"low_cm": "Low-margin products", "distant": "Distant regions", "both": "Both"}
TYPE_COLORS = {TYPE_LABELS[k]: c for k, c in zip(TYPE_LABELS, SERIES)}
RULE_LABELS = {"abc_c": "Weakest products (ABC class C + loss-makers)", "margin_below": "Products with margin below 10%",
               "n/a": "–"}
EPS_LABELS = {"estimated": "Estimated from data", "low": "Customers barely react", "mid": "Customers react moderately",
              "high": "Customers react strongly", "custom": "Custom"}
DIM_LABELS = {"Total": "Total", "Channel": "Sales channel", "CustomerType": "Customer type", "TerritoryGroup": "Region",
              "Territory": "Territory", "Category": "Product category"}
PARETO_DIM_LABELS = {
    "customer": "Customers", "product": "Products", "territory": "Territories", "category": "Product categories",
    "subcategory": "Product subcategories", "country": "Countries", "channel": "Sales channels",
    "customer_type": "Customer types", "segment": "Customer segments", "store_business_type": "Store types",
    "store_size": "Store sizes", "income_band": "Income bands (individuals)", "occupation": "Occupations (individuals)",
    "rfm_segment": "Customer activity (RFM)", "channel_category": "Channel × category"}
ABC_LABELS = {"A": "A – top 80% of profit", "B": "B – next 15%", "C": "C – last 5%", "Negative": "Loss-making"}
ABC_COLORS = {ABC_LABELS["A"]: "#1c5cab", ABC_LABELS["B"]: "#3987e5", ABC_LABELS["C"]: "#86b6ef",
              ABC_LABELS["Negative"]: LOSS}
MODEL_LABELS = {"logistic_regression": "Logistic regression (baseline)", "lightgbm": "LightGBM (machine learning)"}
FEATURE_LABELS = {
    "recency_days": "Days since last order", "frequency": "Number of orders", "monetary": "Total revenue",
    "cm_total": "Total margin", "cm_margin": "Margin % of their purchases", "tenure_days": "Days since first order",
    "avg_order_value": "Average order value", "units": "Units bought", "n_products": "Different products bought",
    "n_subcategories": "Different subcategories bought", "share_bikes": "Share of spend on bikes",
    "share_components": "Share of spend on components", "share_clothing": "Share of spend on clothing",
    "share_accessories": "Share of spend on accessories", "avg_price_index": "Price paid vs list price",
    "discount_share": "Share of spend discounted", "orders_last_90d": "Orders in last 90 days",
    "orders_last_180d": "Orders in last 180 days", "revenue_last_180d": "Revenue in last 180 days",
    "distance_km": "Distance from HQ", "Age": "Age", "TotalChildren": "Children", "NumberCarsOwned": "Cars owned",
    "HomeOwnerFlag": "Home owner", "AnnualSales": "Store annual sales", "NumberEmployees": "Store employees",
    "CustomerType": "Customer type", "Territory": "Territory", "TerritoryGroup": "Region", "IncomeBand": "Income band",
    "Education": "Education", "Occupation": "Occupation", "Gender": "Gender", "MaritalStatus": "Marital status",
    "StoreBusinessType": "Store type", "StoreSize": "Store size", "Specialty": "Store specialty"}

# raw column -> (label, format)
COLS = {
    "revenue": ("Revenue", "money"), "cogs": ("Product cost", "money"), "freight_alloc": ("Shipping cost", "money"),
    "cm": ("Contribution margin", "money"), "cm1": ("Margin before shipping", "money"),
    "cm2": ("Margin after shipping", "money"), "cm1_margin": ("Margin % before shipping", "pct"),
    "cm2_margin": ("Margin % after shipping", "pct"), "cm_margin": ("Margin %", "pct"),
    "revenue_base": ("Revenue before", "money"), "cm_base": ("Margin before", "money"),
    "cm_change": ("Margin change", "money"), "spillover_cm_lost": ("Margin lost via spillover", "money"),
    "affected_revenue_share": ("Revenue marked up", "pct"), "customers_base": ("Customers", "int"),
    "customers_lost": ("Customers lost", "int"), "dimension_value": ("Segment", None),
    "rank": ("Rank", "int"), "entity_name": ("Name", None), "parent": ("Belongs to", None),
    "abc_class": ("ABC class", None), "cm_share": ("Share of total margin", "pct"),
    "cum_positive_cm_share": ("Cumulative share of profit", "pct"), "entities": ("Count", "int"),
    "entity_share": ("Share of count", "pct"), "revenue_share": ("Share of revenue", "pct"),
    "cm_share_of_net": ("Share of total margin", "pct"), "cm_share_of_positive": ("Share of profit", "pct"),
    "top_share": ("Top …", "pct0"), "top_n": ("Number", "int"),
    "Territory": ("Territory", None), "TerritoryGroup": ("Region", None), "Channel": ("Sales channel", None),
    "Category": ("Product category", None), "CustomerType": ("Customer type", None),
    "median_distance_km": ("Typical distance from HQ (km)", "int"),
    "freight_pct_of_revenue": ("Shipping as % of revenue", "pct"),
    "share_distant_territory_group": ("Distant by region group", "pct0"),
    "share_distant_great_circle": ("Distant by km (chosen)", "pct0"),
    "share_distant_freight_per_unit": ("Distant by shipping cost", "pct0"),
    "elasticity_ols": ("Regression estimate", "dec"), "se_ols": ("± uncertainty", "dec"), "p_ols": ("p-value", "dec3"),
    "n_cells": ("Data points", "int"), "elasticity_lgbm": ("Machine-learning estimate", "dec"),
    "plausible": ("Believable?", None), "elasticity_estimated": ("Value used", "dec"),
    "estimated_source": ("Source of value used", None), "cells": ("Data points", "int"),
    "sd_within_log_price": ("Typical price movement", "pct"),
    "share_cells_with_price_change": ("Share of data with any price change", "pct0"),
    "model": ("Model", None), "segment": ("Customers", None), "roc_auc": ("ROC-AUC", "dec"),
    "pr_auc": ("PR-AUC", "dec"), "brier": ("Brier score", "dec3"), "precision": ("Precision", "pct0"),
    "recall": ("Recall", "pct0"), "f1": ("F1", "dec"), "base_rate": ("Actual churn rate", "pct0"),
    "top_decile_churn_rate": ("Churn rate among top-10% risk", "pct0"),
    "markup_type": ("Markup on", None), "low_cm_rule": ("Low-margin definition", None),
    "elasticity_scenario": ("Customer reaction", None), "recommended_level": ("Recommended markup", "pct0"),
    "recommended_cm_change": ("Extra margin at recommended markup", "money"),
    "recommended_customers_lost_pct": ("Customers lost at recommended markup", "pct"),
    "first_unacceptable_level": ("Loss limit crossed at", "pct0"),
    "check": ("Check", None), "table": ("Table", None), "rows_affected": ("Rows affected", "int"),
    "action": ("What we did", None), "churn_risk_band": ("Churn risk", None), "customers": ("Customers", "int"),
}


def _money(v):
    return "–" if pd.isna(v) else (f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}")


FORMATTERS = {"money": _money, "pct": "{:.1%}", "pct0": "{:.0%}", "int": "{:,.0f}", "dec": "{:.2f}",
              "dec3": "{:.3f}"}


def table(df, cols, rename=None, height=None):
    """Show a DataFrame with plain-language column names and formatted numbers."""
    rename = rename or {}
    d = df[cols].copy()
    labels = {c: rename.get(c, COLS.get(c, (c, None))[0]) for c in cols}
    fmts = {labels[c]: FORMATTERS[COLS[c][1]] for c in cols
            if c in COLS and COLS[c][1] and pd.api.types.is_numeric_dtype(d[c])}
    d = d.rename(columns=labels)
    st.dataframe(d.style.format(fmts, na_rep="–"), hide_index=True, use_container_width=True, height=height)


def money(x):
    sign = "-" if x < 0 else ""
    x = abs(x)
    return f"{sign}${x / 1e6:,.2f}M" if x >= 1e6 else f"{sign}${x / 1e3:,.0f}k"


def pct(x):
    return f"{x:+.1%}"


def style(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="",
                      hoverlabel=dict(namelength=-1))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zerolinecolor="rgba(128,128,128,0.4)")
    return fig


def how_to_read(text):
    st.caption(f"**How to read:** {text}")


st.set_page_config(page_title="AdventureWorks Markup Simulator", layout="wide")


# --- Data -----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading pipeline outputs ...")
def load():
    cfg = load_config()
    proc, out = cfg["paths"]["processed_dir"], cfg["paths"]["output_dir"]
    if not (proc / "fact.parquet").exists():
        return cfg, None
    d = {name: pd.read_parquet(proc / f"{name}.parquet")
         for name in ["fact", "customers", "product_abc", "elasticity", "churn_scores"]}
    for name in ["pareto_abc", "pareto_concentration", "abc_class_summary", "cm_summary", "distance_comparison",
                 "elasticity_price_variation", "churn_metrics", "churn_feature_importance", "scenario_summary",
                 "scenario_optimum", "dim_customer", "cleaning_log"]:
        d[name] = pd.read_csv(out / f"{name}.csv", low_memory=False)
    d["engine"] = scenarios.ScenarioEngine(d["fact"], d["product_abc"], d["elasticity"], d["churn_scores"], cfg)
    return cfg, d


cfg, D = load()
if D is None:
    st.error("No pipeline outputs found. Run `python -m src.pipeline` first (manage.sh → 'Run pipeline').")
    st.stop()
engine = D["engine"]
sc = cfg["scenarios"]
assumed = cfg["elasticity"]["assumed"]
distant_km = cfg["distance"]["distant_km"]


def units_drop(eps):
    """Volume change for a 10% price increase at elasticity eps."""
    return 1.1 ** eps - 1


st.title("AdventureWorks · Margin & Markup Simulator")
st.markdown("**Question:** if we raise prices on low-margin products and/or on orders to distant regions, "
            "how much extra margin do we make, and how many customers do we lose?")
st.caption("Contribution margin = revenue − product cost − shipping cost. "
           f"Simulations are based on the last {sc['baseline_months']} months of sales.")

# --- Sidebar: scenario controls ------------------------------------------------------
with st.sidebar:
    st.header("1 · What do we mark up?")
    mtype = st.radio("Raise prices on", list(TYPE_LABELS), format_func=TYPE_LABELS.get, index=2)
    level = st.slider("By how much?", 0, 50, 10, 1, format="%d%%") / 100
    rule = st.radio("Which products count as low-margin?", ["abc_c", "margin_below"], format_func=RULE_LABELS.get,
                    disabled=mtype == "distant",
                    help="Weakest products = together earn only the last 5% of profit, or lose money. "
                         "The 10% rule also catches the big bikes sold to resellers at cost.")
    st.caption(f"Distant region = shipped more than {distant_km:,} km from HQ in Bothell (Europe & Australia).")

    st.header("2 · How do customers react?")
    eps_choice = st.selectbox(
        "Price sensitivity", ["low", "mid", "high", "estimated", "custom"], index=1,
        format_func=lambda k: EPS_LABELS[k] + (f" ({assumed[k]})" if k in assumed else ""),
        help="Elasticity: how much less customers buy when prices rise. The data could not measure this reliably, "
             "so we test a range of assumed values.")
    eps_custom = st.slider("Custom elasticity", -4.0, 0.0, -1.5, 0.1) if eps_choice == "custom" else None
    eps_show = eps_custom if eps_custom is not None else assumed.get(eps_choice, assumed["mid"])
    st.caption(f"→ a 10% price rise means about **{-units_drop(eps_show):.0%} fewer units** sold.")
    churn_link = st.slider("Share of lost sales from customers who leave completely", 0.0, 1.0,
                           float(sc["churn_link"]), 0.05,
                           help="0 = everyone stays but buys less; 1 = all lost sales come from customers leaving.")
    spill = st.checkbox("Customers who leave stop buying everything (not just the marked-up items)",
                        value=sc["include_spillover"])

    st.header("3 · What is acceptable?")
    limit = st.slider("Max. share of customers we are willing to lose", 0.0, 0.20,
                      float(sc["max_customer_loss_pct"]), 0.01, format="%.2f",
                      help="Measured against the customers we would otherwise expect to keep.")

eps_name = "mid" if eps_choice == "custom" else eps_choice
sim_kwargs = dict(low_cm_rule=rule, elasticity_scenario=eps_name, churn_link=churn_link, include_spillover=spill,
                  elasticity_override=eps_custom)


@st.cache_data(show_spinner=False)
def curve(mtype, sim_items):
    kw = dict(sim_items)
    rows = []
    for lv in np.arange(0, 0.51, 0.01):
        res, dp = engine.simulate(mtype, float(lv), **kw)
        rows.append({"markup_level": lv, **engine.totals(res, dp)})
    return pd.DataFrame(rows)


tabs = st.tabs(["① Scenario simulator", "② Where the margin comes from", "③ Top vs. bottom (Pareto)",
                "④ Distant regions", "⑤ Price sensitivity", "⑥ Customer churn", "⑦ Data quality"])

# --- ① Scenario simulator --------------------------------------------------------------
with tabs[0]:
    res, dp = engine.simulate(mtype, level, **sim_kwargs)
    tot = engine.totals(res, dp)
    cm_change = tot["cm"] - tot["cm_base"]
    target = {"low_cm": "low-margin products", "distant": "orders to distant regions",
              "both": "low-margin products and distant regions"}[mtype]
    st.info(f"Raising prices by **{level:.0%}** on **{target}** changes the yearly margin by "
            f"**{money(cm_change)} ({pct(cm_change / tot['cm_base'])})** and costs about "
            f"**{tot['customers_lost']:,.0f} customers** ({tot['customers_lost_pct']:.1%} of those we would "
            f"otherwise keep).")
    k = st.columns(5)
    k[0].metric("Yearly contribution margin", money(tot["cm"]), pct(cm_change / tot["cm_base"]),
                help="Margin after product and shipping cost, compared with today.")
    k[1].metric("Yearly revenue", money(tot["revenue"]), pct(tot["revenue"] / tot["revenue_base"] - 1))
    k[2].metric("Units sold", f"{tot['units']:,.0f}", pct(tot["units"] / tot["units_base"] - 1))
    k[3].metric("Customers lost", f"{tot['customers_lost']:,.0f}", f"{tot['customers_lost_pct']:.1%}",
                delta_color="inverse", help="Expected extra customers who stop buying because of the markup.")
    k[4].metric("Revenue affected by markup", f"{np.average(res.affected, weights=res.revenue_base):.0%}")
    if tot["customers_lost_pct"] > limit:
        st.warning(f"⚠️ Customer loss of {tot['customers_lost_pct']:.1%} is above your limit of {limit:.0%}.")

    cv = curve(mtype, tuple(sim_kwargs.items()))
    cv["cm_change"] = cv.cm - cv.cm_base
    ok = cv[cv.customers_lost_pct <= limit]
    best_ok = ok.loc[ok.cm.idxmax()] if len(ok) else None
    breach = cv[cv.customers_lost_pct > limit].markup_level.min()

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Scatter(x=cv.markup_level, y=cv.cm_change, mode="lines",
                                   line=dict(width=2, color=SERIES[0]), name="Extra margin",
                                   hovertemplate="Markup %{x:.0%}<br>Extra margin $%{y:,.0f}<extra></extra>"))
        fig.add_vline(x=level, line_dash="dot", line_color="gray", annotation_text="your choice",
                      annotation_position="top left")
        if best_ok is not None:
            fig.add_trace(go.Scatter(x=[best_ok.markup_level], y=[best_ok.cm_change], mode="markers+text",
                                     marker=dict(size=11, color=SERIES[2]), name="Best within limit",
                                     text=[f"best: {best_ok.markup_level:.0%}"], textposition="top left",
                                     hovertemplate="Best markup within the customer limit: %{x:.0%}<extra></extra>"))
        fig.update_layout(title="Extra margin per year at each markup level", showlegend=False)
        fig.update_xaxes(tickformat=".0%", title="Markup level")
        fig.update_yaxes(title="Extra margin ($ per year)", tickprefix="$")
        st.plotly_chart(style(fig), use_container_width=True)
        how_to_read("the higher the line, the more extra margin. Dotted line = your selected markup. "
                    "Green dot = the highest margin that stays within your customer-loss limit.")
    with c2:
        fig = go.Figure(go.Scatter(x=cv.markup_level, y=cv.customers_lost_pct, mode="lines",
                                   line=dict(width=2, color=SERIES[1]), name="Customers lost",
                                   hovertemplate="Markup %{x:.0%}<br>Customers lost %{y:.1%}<extra></extra>"))
        fig.add_hline(y=limit, line_dash="dash", line_color=LOSS, annotation_text=f"your limit ({limit:.0%})",
                      annotation_position="top left")
        fig.add_vline(x=level, line_dash="dot", line_color="gray")
        fig.update_layout(title="Share of customers lost at each markup level", showlegend=False)
        fig.update_xaxes(tickformat=".0%", title="Markup level")
        fig.update_yaxes(tickformat=".0%", title="Customers lost (% of expected customers)")
        st.plotly_chart(style(fig), use_container_width=True)
        how_to_read("where the orange line crosses the red dashed line, customer loss becomes unacceptable"
                    + (f" — here at a markup of **{breach:.0%}**." if pd.notna(breach)
                       else " — with these settings it never does."))

    st.subheader("Where does the extra margin come from?")
    seg = engine.aggregate(res, dp)
    dim = st.selectbox("Split by", ["Territory", "TerritoryGroup", "Category", "Channel", "CustomerType"],
                       format_func=DIM_LABELS.get)
    s = seg[seg.dimension == dim].copy()
    s["cm_change"] = s.cm - s.cm_base
    s["Direction"] = np.where(s.cm_change >= 0, "Gain", "Loss")
    fig = px.bar(s.sort_values("cm_change"), x="cm_change", y="dimension_value", orientation="h", color="Direction",
                 color_discrete_map={"Gain": SERIES[0], "Loss": LOSS},
                 labels={"cm_change": "Change in yearly margin ($)", "dimension_value": ""},
                 hover_data={"Direction": False, "cm_base": ":$,.0f", "cm": ":$,.0f",
                             "affected_revenue_share": ":.0%"})
    fig.update_layout(title=f"Change in yearly margin by {DIM_LABELS[dim].lower()}", bargap=0.35, showlegend=False)
    st.plotly_chart(style(fig, 340), use_container_width=True)
    how_to_read("longer bars = more extra margin from that segment. Segments with no marked-up sales show ~0.")
    cols = ["dimension_value", "revenue_base", "revenue", "cm_base", "cm", "cm_change", "affected_revenue_share",
            "spillover_cm_lost"] + (["customers_base", "customers_lost"] if s.customers_lost.notna().any() else [])
    table(s.sort_values("cm_change", ascending=False), cols,
          rename={"dimension_value": DIM_LABELS[dim], "revenue": "Revenue after", "cm": "Margin after"})

    with st.expander("Compare all three markup types (pre-calculated 0–20%, same data as the Power BI files)"):
        g = D["scenario_summary"]
        g = g[(g.elasticity_scenario == eps_name) & g.low_cm_rule.isin([rule, "n/a"])].copy()
        g["Markup on"] = g.markup_type.map(TYPE_LABELS)
        fig = px.line(g, x="markup_level", y="cm_change", color="Markup on", markers=True,
                      color_discrete_map=TYPE_COLORS,
                      labels={"cm_change": "Extra margin ($ per year)", "markup_level": "Markup level"})
        fig.update_xaxes(tickformat=".0%")
        fig.update_layout(title=f"Extra margin by markup type ({EPS_LABELS[eps_name].lower()})")
        st.plotly_chart(style(fig, 320), use_container_width=True)
        opt = D["scenario_optimum"].copy()
        opt = opt[opt.elasticity_scenario == eps_name]
        opt["markup_type"] = opt.markup_type.map(TYPE_LABELS)
        opt["low_cm_rule"] = opt.low_cm_rule.map(RULE_LABELS)
        opt["elasticity_scenario"] = opt.elasticity_scenario.map(EPS_LABELS)
        table(opt, ["markup_type", "low_cm_rule", "recommended_level", "recommended_cm_change",
                    "recommended_customers_lost_pct", "first_unacceptable_level"])
        st.caption(f"Recommended = highest margin with at most {sc['max_customer_loss_pct']:.0%} customers lost "
                   "(limit fixed in the pre-calculated files).")

# --- ② Where the margin comes from ------------------------------------------------------
with tabs[1]:
    f = D["fact"]
    cms = D["cm_summary"].set_index("Channel")
    st.markdown("Revenue, cost and margin for the whole period (May 2011 – June 2014).")
    k = st.columns(4)
    k[0].metric("Revenue", money(cms.loc["Total", "revenue"]))
    k[1].metric("Margin before shipping", money(cms.loc["Total", "cm1"]),
                f"{cms.loc['Total', 'cm1_margin']:.1%} of revenue", delta_color="off")
    k[2].metric("Margin after shipping", money(cms.loc["Total", "cm2"]),
                f"{cms.loc['Total', 'cm2_margin']:.1%} of revenue", delta_color="off")
    k[3].metric("Reseller margin after shipping", money(cms.loc["Reseller", "cm2"]),
                f"{cms.loc['Reseller', 'cm2_margin']:.1%} of revenue", delta_color="off")

    m = f.groupby(["OrderMonth", "Channel"])[["revenue", "cm"]].sum().reset_index()
    channel_colors = {"Online": SERIES[0], "Reseller": SERIES[1]}
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(m, x="OrderMonth", y="revenue", color="Channel", color_discrete_map=channel_colors,
                      labels={"revenue": "Revenue ($)", "OrderMonth": ""})
        fig.update_traces(line_width=2)
        fig.update_layout(title="Monthly revenue: resellers bring most of the revenue", hovermode="x unified")
        st.plotly_chart(style(fig), use_container_width=True)
    with c2:
        fig = px.line(m, x="OrderMonth", y="cm", color="Channel", color_discrete_map=channel_colors,
                      labels={"cm": "Contribution margin ($)", "OrderMonth": ""})
        fig.update_traces(line_width=2)
        fig.add_hline(y=0, line_color="gray", line_width=1)
        fig.update_layout(title="Monthly margin: online earns almost all of it", hovermode="x unified")
        st.plotly_chart(style(fig), use_container_width=True)
    how_to_read("compare the two charts — the reseller line is high in revenue but sits around zero in margin.")

    cc = f.groupby(["Channel", "Category"])[["revenue", "cm"]].sum().reset_index()
    cc["cm_margin"] = cc.cm / cc.revenue
    fig = px.bar(cc, x="Category", y="cm", color="Channel", barmode="group", color_discrete_map=channel_colors,
                 hover_data={"revenue": ":$,.0f", "cm_margin": ":.1%"},
                 labels={"cm": "Contribution margin ($)", "Category": ""})
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.update_layout(title="Margin by product category: bikes sold to resellers lose money", bargap=0.3,
                      bargroupgap=0.08)
    st.plotly_chart(style(fig), use_container_width=True)
    how_to_read("bars below zero lose money. Resellers pay about 60% of list price — for bikes that is roughly "
                "what the bike costs us.")
    table(cms.reset_index(), ["Channel", "revenue", "cogs", "freight_alloc", "cm1", "cm2", "cm1_margin", "cm2_margin"])

# --- ③ Pareto / ABC ---------------------------------------------------------------------
with tabs[2]:
    st.markdown("Is it true that a few customers, products and regions earn most of the margin? "
                "Everything is ranked from most to least profitable and grouped into classes: "
                "**A** = the top earners that together make 80% of the profit, **B** = the next 15%, "
                "**C** = the last 5%, plus a separate group for **loss-makers**.")
    pa = D["pareto_abc"]
    dims = [d for d in PARETO_DIM_LABELS if d in pa.dimension.unique()]
    dim = st.selectbox("Look at", dims, format_func=PARETO_DIM_LABELS.get)
    name = PARETO_DIM_LABELS[dim].lower()
    g = pa[pa.dimension == dim].sort_values("rank").copy()
    g["ABC class"] = g.abc_class.map(ABC_LABELS)
    conc = D["pareto_concentration"]
    c1, c2 = st.columns([2, 1])
    with c1:
        if len(g) > 40:
            top20 = conc[(conc.dimension == dim) & (conc.top_share == 0.2)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(width=1, dash="dot", color="gray"),
                                     name="If everyone earned the same", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=g.entity_share, y=g.cum_positive_cm_share, mode="lines",
                                     line=dict(width=2, color=SERIES[0]), name="Actual",
                                     hovertemplate="Top %{x:.0%} of " + name + "<br>earn %{y:.0%} of the profit"
                                                   "<extra></extra>"))
            if len(top20):
                y20 = top20.cm_share_of_positive.iloc[0]
                fig.add_trace(go.Scatter(x=[0.2], y=[y20], mode="markers+text", marker=dict(size=10, color=SERIES[1]),
                                         text=[f"top 20% → {y20:.0%} of profit"], textposition="bottom right",
                                         showlegend=False, hoverinfo="skip"))
            fig.update_xaxes(tickformat=".0%", title=f"Share of {name} (most profitable first)")
            fig.update_yaxes(tickformat=".0%", title="Share of total profit earned")
            fig.update_layout(title=f"How concentrated is the profit across {name}?",
                              legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(style(fig, 440), use_container_width=True)
            how_to_read("the further the blue line bulges above the dotted line, the more the profit is "
                        "concentrated in a few top earners. Loss-makers are left out of this curve and counted "
                        "separately in the table on the right.")
        else:
            fig = px.bar(g, x="entity_name", y="cm", color="ABC class", color_discrete_map=ABC_COLORS,
                         category_orders={"ABC class": list(ABC_LABELS.values())},
                         labels={"entity_name": "", "cm": "Contribution margin ($)"},
                         hover_data={"revenue": ":$,.0f", "cm_margin": ":.1%"})
            fig.add_hline(y=0, line_color="gray", line_width=1)
            fig.update_layout(title=f"Margin by {name}, most profitable first", bargap=0.3)
            st.plotly_chart(style(fig, 440), use_container_width=True)
            how_to_read("bar height = margin earned over the whole period; colour = ABC class; red bars lose money.")
    with c2:
        st.markdown("**Size of each class**")
        cs = D["abc_class_summary"]
        cs = cs[cs.dimension == dim].copy()
        cs["abc_class"] = cs.abc_class.map(ABC_LABELS)
        table(cs, ["abc_class", "entities", "entity_share", "revenue_share", "cm"],
              rename={"entities": f"Number of {name}", "entity_share": f"Share of {name}"})
        if dim in conc.dimension.values:
            st.markdown("**Profit earned by the top …**")
            cc = conc[conc.dimension == dim].copy()
            cc["top_share"] = cc.top_share.map(lambda v: f"Top {v:.0%} of {name}")
            table(cc, ["top_share", "top_n", "revenue_share", "cm_share_of_positive"],
                  rename={"top_share": "Group", "cm_share_of_positive": "Share of profit"})
    st.markdown(f"**All {name}, ranked** (first 500)")
    table(g.head(500), ["rank", "entity_name", "parent", "revenue", "cm", "cm_margin", "cum_positive_cm_share",
                        "ABC class"], rename={"ABC class": "ABC class"})

# --- ④ Distance ---------------------------------------------------------------------------
with tabs[3]:
    st.markdown(f"We call an order **distant** when it is shipped more than **{distant_km:,} km** (straight-line) "
                "from headquarters in Bothell, Washington. In practice that means Europe and Australia.")
    f = D["fact"]
    city = f.groupby(["City", "ShipCountry", "is_distant"]).agg(
        lat=("Latitude", "median"), lon=("Longitude", "median"), revenue=("revenue", "sum"), cm=("cm", "sum"),
        distance_km=("distance_km", "median")).reset_index()
    near_lbl, far_lbl = f"Near (≤ {distant_km:,} km)", f"Distant (> {distant_km:,} km)"
    city["Region"] = np.where(city.is_distant, far_lbl, near_lbl)
    fig = px.scatter_geo(city, lat="lat", lon="lon", size="revenue", color="Region", size_max=25,
                         color_discrete_map={near_lbl: SERIES[0], far_lbl: SERIES[1]}, hover_name="City",
                         labels={"ShipCountry": "Country", "revenue": "Revenue", "cm": "Margin",
                                 "distance_km": "Distance from HQ (km)"},
                         hover_data={"ShipCountry": True, "revenue": ":$,.0f", "cm": ":$,.0f", "distance_km": ":,.0f",
                                     "lat": False, "lon": False, "Region": False})
    fig.add_trace(go.Scattergeo(lat=[cfg["distance"]["hq_lat"]], lon=[cfg["distance"]["hq_lon"]],
                                mode="markers+text", text=["HQ Bothell"], textposition="top center", name="HQ",
                                marker=dict(size=12, symbol="star", color="#0b0b0b")))
    fig.update_geos(showcountries=True, countrycolor="rgba(128,128,128,0.3)", projection_type="natural earth")
    fig.update_layout(title="Where our customers are (bubble size = revenue)")
    st.plotly_chart(style(fig, 480), use_container_width=True)

    dc = D["distance_comparison"].sort_values("median_distance_km")
    fig = px.bar(dc, x="Territory", y="freight_pct_of_revenue", color_discrete_sequence=[SERIES[0]],
                 hover_data={"median_distance_km": ":,.0f"},
                 labels={"freight_pct_of_revenue": "Shipping cost as % of revenue", "Territory": "",
                         "median_distance_km": "Distance from HQ (km)"})
    fig.update_yaxes(tickformat=".1%", rangemode="tozero")
    fig.update_layout(title="Shipping cost does not rise with distance (territories sorted near → far)", bargap=0.35)
    st.plotly_chart(style(fig, 340), use_container_width=True)
    how_to_read("all bars are about the same height: shipping is charged as a fixed ~3% of the order value, no "
                "matter how far it goes. A markup on distant regions is therefore a pricing decision, not a way to "
                "recover higher shipping costs.")

    st.markdown("**Three ways to define 'distant' — share of each territory's revenue that would count as distant**")
    table(dc, ["TerritoryGroup", "Territory", "median_distance_km", "share_distant_territory_group",
               "share_distant_great_circle", "share_distant_freight_per_unit"])
    st.caption("We use straight-line distance: it uses the real delivery address and can be adjusted. Region groups "
               "are too coarse, and shipping cost flags customers almost at random.")

# --- ⑤ Price sensitivity --------------------------------------------------------------------
with tabs[4]:
    e = D["elasticity"].copy()
    n_ok, n_all = int(e.plausible.sum()), len(e)
    st.markdown("**Price sensitivity (elasticity)** = how much less customers buy when the price goes up. "
                "We tried to measure it from past price changes and discounts.")
    st.warning(f"**Result: only {n_ok} of {n_all} estimates are believable.** Online prices almost never changed, "
               "and reseller prices are lower for bigger orders, which confuses the models. The simulator therefore "
               "uses assumed values and shows the result for each.")
    a = pd.DataFrame({"Customer reaction": [EPS_LABELS[k] for k in ["low", "mid", "high"]],
                      "Elasticity": [assumed[k] for k in ["low", "mid", "high"]]})
    a["Units lost after a 10% price rise"] = [f"{-units_drop(v):.0%}" for v in a.Elasticity]
    st.dataframe(a, hide_index=True)

    e["cell"] = e.Channel + " · " + e.Category + " · " + e.TerritoryGroup
    long = e.melt(id_vars=["cell"], value_vars=["elasticity_ols", "elasticity_lgbm"], var_name="Model",
                  value_name="Estimate")
    long["Model"] = long.Model.map({"elasticity_ols": "Statistical regression", "elasticity_lgbm": "Machine learning"})
    fig = px.scatter(long, x="Estimate", y="cell", color="Model", symbol="Model",
                     color_discrete_sequence=SERIES[:2], labels={"cell": "", "Estimate": "Estimated elasticity"})
    lo, hi = cfg["elasticity"]["plausible_range"]
    fig.add_vrect(x0=lo, x1=hi, fillcolor="gray", opacity=0.12, line_width=0, annotation_text="believable range",
                  annotation_position="top left")
    fig.add_vline(x=0, line_color="gray", line_width=1)
    fig.update_traces(marker_size=10)
    fig.update_layout(title="Estimated price sensitivity per channel, product category and region",
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(style(fig, 540), use_container_width=True)
    how_to_read("each row is one channel × category × region. Points inside the grey band would be realistic "
                "(customers buy less when prices rise). Points to the right of 0 would mean customers buy *more* "
                "when prices rise — not believable. The two models also disagree strongly with each other.")
    e["plausible"] = e.plausible.map({True: "Yes", False: "No"})
    table(e, ["Channel", "Category", "TerritoryGroup", "elasticity_ols", "p_ols", "n_cells", "elasticity_lgbm",
              "plausible", "elasticity_estimated", "estimated_source"])
    st.markdown("**How much did prices actually move?**")
    table(D["elasticity_price_variation"], ["Channel", "cells", "sd_within_log_price", "share_cells_with_price_change"])

# --- ⑥ Churn --------------------------------------------------------------------------------
with tabs[5]:
    st.markdown("A machine-learning model predicts, for every customer, the chance that they **stop buying** "
                f"(no order in the next {cfg['churn']['window_months']} months). The simulator uses it as each "
                "customer's starting risk; a markup then pushes that risk up.")
    met = D["churn_metrics"].copy()
    best = met[(met.segment == "All")].sort_values("roc_auc", ascending=False).iloc[0]
    k = st.columns(3)
    k[0].metric("Model accuracy (ROC-AUC)", f"{best.roc_auc:.2f}", help="0.5 = coin flip, 1.0 = perfect.")
    k[1].metric("Customers who actually stopped (test period)", f"{best.base_rate:.0%}")
    k[2].metric("…among the 10% the model flagged as riskiest", f"{best.top_decile_churn_rate:.0%}")
    st.caption(f"Tested honestly on the future: trained on data up to {cfg['churn']['validation_snapshot_date']}, "
               f"checked against what happened after {cfg['churn']['snapshot_date']}.")

    c1, c2 = st.columns(2)
    with c1:
        fi = D["churn_feature_importance"].head(12).copy()
        fi["feature"] = fi.feature.map(lambda c: FEATURE_LABELS.get(c, c))
        fig = px.bar(fi.sort_values("permutation_importance_auc"), x="permutation_importance_auc", y="feature",
                     orientation="h", color_discrete_sequence=[SERIES[0]],
                     labels={"permutation_importance_auc": "Importance (drop in accuracy without it)", "feature": ""})
        fig.update_layout(title="What best predicts that a customer stops buying", bargap=0.35)
        st.plotly_chart(style(fig, 440), use_container_width=True)
        how_to_read("longer bar = the model relies more on this information.")
    with c2:
        dc = D["dim_customer"]
        bands = ["Low (<30%)", "Medium (30-60%)", "High (60-80%)", "Very high (>80%)"]
        rb = dc.groupby(["churn_risk_band", "CustomerType"]).agg(customers=("CustomerID", "size"),
                                                                 cm=("cm", "sum")).reset_index()
        fig = px.bar(rb, x="churn_risk_band", y="customers", color="CustomerType", barmode="group",
                     category_orders={"churn_risk_band": bands}, color_discrete_sequence=SERIES[:2],
                     labels={"churn_risk_band": "Chance of stopping in the next 6 months", "customers": "Customers",
                             "CustomerType": "Customer type"})
        fig.update_layout(title="Customers by churn risk today", bargap=0.3, bargroupgap=0.08)
        st.plotly_chart(style(fig, 440), use_container_width=True)
        how_to_read("most individual customers buy once and are unlikely to return; most stores are loyal.")

    met["model"] = met.model.map(MODEL_LABELS)
    met["segment"] = met.segment.map({"All": "All", "Individual": "Individuals", "Store": "Stores"})
    table(met, ["model", "segment", "roc_auc", "precision", "recall", "base_rate", "top_decile_churn_rate"])
    st.caption("ROC-AUC: 0.5 = coin flip, 1 = perfect. Precision: of the customers flagged as leaving, how many "
               "really left. Recall: of the customers who left, how many the model caught.")

# --- ⑦ Data quality --------------------------------------------------------------------------
with tabs[6]:
    st.markdown("Every check we ran on the raw data, and what we did about it.")
    log = D["cleaning_log"]
    if st.checkbox("Only show checks that found something", value=True):
        log = log[log.rows_affected > 0]
    table(log, ["check", "table", "rows_affected", "action"])
