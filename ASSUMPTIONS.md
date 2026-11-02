# Modelling assumptions

Every assumption the analysis relies on, with the reason and where it can be changed. Settings refer to
`config.yaml`. Numbers are from AdventureWorks2022 (orders 2011-05-31 to 2014-06-30).

## Data & cleaning

| # | Assumption | Why | Setting / code |
|---|---|---|---|
| D1 | AdventureWorks2022 **OLTP** is the data source (not the DW). | It contains freight per order, discounts/special offers, store vs. individual customers and ship-to coordinates, which the brief needs. | `src/extract.py` |
| D2 | All amounts are in USD; tax (`TaxAmt`) is excluded from revenue. | AdventureWorks stores amounts in USD; tax is passed through to authorities. | `src/cm.py` |
| D3 | `LineTotal` is the source of truth for line revenue. | It matches UnitPrice × Qty × (1 − discount) to within rounding ($0.06 in total). | `src/cleaning.py` |
| D4 | Cost/list price history `EndDate` is **inclusive**. | Periods run e.g. 2011-05-31 → 2012-05-29, next starts 2012-05-30; orders dated on the end date would otherwise have no cost. | `src/cm.py` |
| D5 | Lines sold after a product's last cost period (64 reseller lines, discontinued stock) use the **last known cost**. | Clearance sales of discontinued products; flagged in `cost_within_period`. | `src/cm.py` |
| D6 | Addresses at (0, 0) (918, France/Germany/UK) get the median coordinates of the same city (else same state). | "Null island" placeholders; the city is known. | `src/cleaning.py` |
| D7 | Reseller data ends 2014-05-01, online data 2014-06-30: recency, baselines and churn scoring are measured **per channel** against that channel's last order date. | Otherwise resellers would look two months less active. | `src/segments.py`, `src/churn.py`, `src/scenarios.py` |
| D8 | Store survey code `BusinessType` is decoded as BM = Value Added Reseller, BS = Specialty Bike Shop, OS = Warehouse. | Same decoding as in AdventureWorksDW `DimReseller`. | `src/segments.py` |

## Contribution margin

| # | Assumption | Why | Setting |
|---|---|---|---|
| C1 | COGS = OrderQty × **standard cost valid at the order date**, not today's `Product.StandardCost`. | Costs changed yearly; using current cost understates reseller CM by ≈ $2.8M. | `cm.cost_basis: historical` |
| C2 | Freight is treated as a cost of the company and allocated to lines **proportionally to line revenue** (CM2). CM1 (before freight) is always reported too. | Brief asks for the optional freight allocation. Note: `TotalDue` includes `Freight`, i.e. freight may be billed to the customer — then CM1 is the better measure. | `cm.freight_allocation: true` |
| C3 | CM is a *contribution* margin: no fixed costs, sales commissions or overheads. | Not available in the data. | — |

## Pareto / ABC

| # | Assumption | Why | Setting |
|---|---|---|---|
| A1 | Classes on cumulative share of **positive** CM: A ≤ 80%, B ≤ 95%, C rest; CM < 0 → class `Negative`. | With loss-making entities the net cumulative curve exceeds 100%, which would push most entities into C. Net shares are reported alongside. | `abc.a_threshold`, `abc.b_threshold` |
| A2 | ABC is computed on the full period (2011-05 to 2014-06). | Most stable view; scenarios use the latest 12 months. | `src/pareto.py` |
| A3 | Derived segments: channel/customer type, store business type and size, individual income band and occupation, RFM segment (recency ≤ 180 days & ≥ 3 orders = loyal, etc.). | Interpretable segments from available attributes. | `src/segments.py` |

## Distance

| # | Assumption | Why | Setting |
|---|---|---|---|
| G1 | "Distant" = **great-circle distance from HQ (Bothell, WA, 47.76 N / 122.21 W) to the ship-to address > 5,000 km**. | Uses the actual delivery address, is continuous (bands, threshold sensitivity). At 5,000 km it equals Europe + Pacific. | `distance.distant_km`, `distance.hq_lat/lon` |
| G2 | Territory groups rejected as primary definition. | Too coarse: Canada (incl. Toronto) counts as near, US East Coast (≈ 3,900 km) the same as Seattle. | — |
| G3 | Freight per unit rejected as a proxy. | Freight is a flat ≈ 2.5% (online) / ≈ 3% (reseller) of order value; correlation with distance ≈ −0.05. A distant-region markup is therefore price discrimination, not cost recovery. | — |

## Price sensitivity

| # | Assumption | Why | Setting |
|---|---|---|---|
| E1 | Demand response is estimated on a product × territory × channel × month panel, with product × territory and month fixed effects. | Identifies elasticity from price changes *within* a product over time, not from cheap vs. expensive products. | `src/elasticity.py` |
| E2 | Lines with **volume discounts are excluded**. | The discount is triggered by the quantity → reverse causality. | `elasticity.exclude_volume_discounts` |
| E3 | An estimate is used only if significant (p < 0.05), within **[−4, −0.2]** and based on ≥ 200 cells; otherwise the cell uses the assumed mid value. | Guards against implausible / wrong-sign estimates. | `elasticity.plausible_range`, `elasticity.min_obs` |
| E4 | **The data cannot support credible elasticities** (1 of 15 cells passes). Scenarios therefore use **assumed elasticities low −0.5 / mid −1.5 / high −2.5** as a sensitivity parameter. | Online prices only change at 3 yearly list-price updates that coincide with assortment changes; reseller prices are tiered by quantity (LightGBM learns reverse causality); OLS gives ≈ 0 or positive values for resellers. The range brackets typical estimates for durable consumer goods / B2B. | `elasticity.assumed` |
| E5 | Constant elasticity per channel × category × territory group: volume × (1 + markup)^ε. | Standard, transparent; a local approximation — markups far beyond 20% are extrapolation. | `src/scenarios.py` |
| E6 | Online demand = number of order lines (units), since online quantity is always 1. | No quantity variation online. | `src/elasticity.py` |

## Churn

| # | Assumption | Why | Setting |
|---|---|---|---|
| H1 | Churn = customer bought on or before the snapshot and places **no order in the following 6 months**. | Resellers order every ~100 days, so 6 months is ≈ 2 missed cycles; for online customers it is a reasonable re-purchase window given 3 years of data. | `churn.window_months` |
| H2 | Out-of-time evaluation: train on snapshot **2013-06-30**, test on **2013-12-31**; features only use data up to the snapshot. | Mimics real use (predict the future from the past) and tests robustness across the mid-2013 break. | `churn.validation_snapshot_date`, `churn.snapshot_date` |
| H3 | For resellers the 2014-H1 label window effectively ends 2014-05-01 (end of their data). | Covers at least one ordering cycle (reseller orders are clustered in Mar and May 2014). | — |
| H4 | The final model (best ROC-AUC, LightGBM) is refit on both snapshots and scores every customer at the end of the data → baseline probability p₀ of not buying in the next 6 months. | Uses all labelled history. | `src/churn.py` |
| H5 | Demographic fields that may leak future information (`TotalPurchaseYTD`, `DateFirstPurchase`) are not used. | Computed outside the snapshot logic. | `src/extract.py` |

## Scenario engine

| # | Assumption | Why | Setting |
|---|---|---|---|
| S1 | Baseline = **last 12 months per channel** (online 2013-07 to 2014-06, reseller 2013-05 to 2014-05), treated as one year of business. | Most recent, complete year; includes the post-2013 assortment. | `scenarios.baseline_months` |
| S2 | Markup multiplies the price of affected lines by (1 + m). A line that is both low-CM and distant (type `both`) gets **both** markups: (1 + m)². | Two separate surcharges. | `src/scenarios.py` |
| S3 | Low-CM product rules: `abc_c` = product ABC class C or Negative; `margin_below` = product CM margin < 10%. Product classification is computed over all channels. | Brief asks for a configurable definition; both are shown because they select very different products (31% vs. 61% of revenue). | `scenarios.low_cm_rules`, `scenarios.low_cm_margin_threshold` |
| S4 | Freight scales with order value. | Freight is a fixed % of order value in the data. | — |
| S5 | **Churn uplift**: Δp_c = (1 − p₀_c) · κ · v_c, where v_c is the revenue-weighted volume loss on the customer's marked-up purchases and **κ = 0.5** the share of that loss caused by customers leaving entirely (rest = buying less). | The churn model has no price response (no customer ever saw a price increase), so the link between markup and churn must be assumed. κ is a sensitivity parameter (dashboard slider). | `scenarios.churn_link` |
| S6 | **Spillover**: a customer lost due to the markup also stops buying the non-marked-up products (expected value Δp_c × their CM on those products). | Losing a customer loses the whole relationship. | `scenarios.include_spillover` |
| S7 | **Unacceptable customer loss** = more than **5% of the customers expected to stay active** (Σ(1 − p₀)). | Management threshold to be agreed; dashboard slider. | `scenarios.max_customer_loss_pct` |
| S8 | Grid 0–20% in 2-point steps; the dashboard extends to 50%. | Brief range; profit keeps rising to 20% in all scenarios because reseller prices are near cost. | `scenarios.markup_levels` |
| S9 | Not modelled: competitor reactions, substitution between products, reseller contract renegotiation, changes in fixed costs. | No data. | — |
