# Output schemas (`outputs/`)

All files are UTF-8 CSV with a header row, comma separator, **`.` as decimal separator**, dates as `YYYY-MM-DD`
and booleans as `True`/`False`. Money is in USD. Regenerate with `docker/manage.sh` → *Run pipeline*.

## Loading into Power BI

1. *Get Data → Text/CSV*, pick the file, then **Transform Data**.
2. On Windows with German/Austrian regional settings, set the locale before changing types:
   *Home → Data type* is not enough. Use *Transform → Data Type → Using Locale… → English (United States)* for
   decimal columns, or set *File → Options → Current file → Regional settings → Locale for import = English (United States)*.
   Otherwise `0.386` is read as `386`.
3. Suggested model (star schema):

   ```
   dim_product (ProductID) 1 ──── * fact_order_lines * ──── 1 dim_customer (CustomerID)
   ```

   `pareto_abc`, `scenario_results`, `scenario_summary`, `scenario_optimum` and the remaining tables are standalone.
4. Scenario page: put `markup_type`, `low_cm_rule`, `elasticity_scenario`, `markup_level` and `dimension` from
   `scenario_results` on slicers (single select for `dimension`), then plot `cm_change` / `customers_lost_pct`
   against `markup_level` or `dimension_value`. Filter `dimension = "Total"` for headline KPIs.

---

## fact_order_lines.csv — one row per order line (121,317)

| Column | Type | Description |
|---|---|---|
| SalesOrderID, SalesOrderDetailID | int | order and line keys (line key is unique) |
| OrderDate, OrderMonth | date | order date; first day of the order month |
| Year, Quarter, Month | int | calendar parts of OrderDate |
| CustomerID | int | → dim_customer |
| CustomerType | text | `Individual` or `Store` |
| Channel | text | `Online` (individuals) or `Reseller` (stores) |
| ProductID, Product, Subcategory, Category | int/text | → dim_product; product hierarchy |
| SpecialOfferID, OfferType | int/text | special offer applied (`No Discount`, `Volume Discount`, `Seasonal Discount`, …) |
| OrderQty | int | units |
| UnitPrice, UnitPriceDiscount | dec | price charged per unit; discount fraction |
| list_price | dec | list price valid at the order date |
| effective_price | dec | UnitPrice × (1 − UnitPriceDiscount) |
| price_index | dec | effective_price / list_price (1.0 online, ≈0.6 reseller) |
| gross_revenue | dec | UnitPrice × OrderQty |
| discount_amount | dec | gross_revenue − revenue |
| revenue | dec | LineTotal = UnitPrice × OrderQty × (1 − UnitPriceDiscount) |
| unit_cost | dec | standard cost valid at the order date (last known cost for discontinued stock) |
| cogs | dec | unit_cost × OrderQty |
| freight_alloc | dec | order freight allocated by line revenue share |
| cm1 | dec | revenue − cogs |
| cm2 | dec | cm1 − freight_alloc |
| cm | dec | the CM used in all analyses (cm2 if `freight_allocation: true`, else cm1) |
| cm_margin | dec | cm / revenue |
| cost_within_period | bool | False = sold after the product's last cost period (last known cost used) |
| TerritoryID, Territory, TerritoryGroup | int/text | sales territory and group (North America / Europe / Pacific) |
| ShipCountry, StateProvince, City | text | ship-to address |
| Latitude, Longitude | dec | ship-to coordinates ((0,0) values imputed, see cleaning_log) |
| distance_km | dec | great-circle distance from HQ (Bothell, WA) |
| distance_band | text | `0-1,000 km`, `1,000-2,500 km`, `2,500-5,000 km`, `5,000-10,000 km`, `10,000-20,000 km` |
| is_distant | bool | distance_km > `distance.distant_km` (default 5,000) |

## dim_product.csv — one row per product sold (266)

| Column | Type | Description |
|---|---|---|
| ProductID, Product, Subcategory, Category | int/text | product hierarchy |
| revenue, cogs, cm, lines | dec/int | totals over the full period |
| cm_margin | dec | cm / revenue |
| rank | int | rank by CM (1 = highest) |
| entity_share | dec | rank / number of products |
| cm_share, cum_cm_share | dec | share and cumulative share of **net** total CM |
| cum_positive_cm_share | dec | cumulative share of **positive** CM (basis of the classes) |
| abc_class_cm | text | `A` (first 80% of positive CM), `B` (next 15%), `C` (rest ≥ 0), `Negative` (CM < 0) |
| low_cm_abc_c | bool | low-CM under rule `abc_c` (class C or Negative) |
| low_cm_margin_below | bool | low-CM under rule `margin_below` (cm_margin < 10%) |

## dim_customer.csv — one row per buying customer (19,119)

| Column | Type | Description |
|---|---|---|
| CustomerID | int | key |
| first_order, last_order | date | first and last order date |
| orders, units | int | number of orders; units bought |
| revenue, cm, cm1, cm_margin | dec | totals over the full period |
| Channel, CustomerType | text | Online/Individual or Reseller/Store |
| Territory, TerritoryGroup, ShipCountry | text | territory of the customer's orders |
| distance_km, is_distant | dec/bool | median ship-to distance from HQ; distant flag |
| recency_days, tenure_days | int | days since last / first order, measured at the end of the customer's channel data |
| RFMSegment | text | `Loyal active` (≤180 days & ≥3 orders), `Active (1-2 orders)`, `Cooling (6-12 months)`, `Lapsed (>12 months)` |
| PersonID, StoreID, StoreName | int/text | source keys |
| StoreBusinessType | text | `Value Added Reseller`, `Specialty Bike Shop`, `Warehouse`; `n/a (individual)` |
| StoreSize | text | by employees: `Small (<=10)`, `Medium (11-50)`, `Large (>50)` |
| Specialty, AnnualSales, NumberEmployees | text/dec | store survey |
| IncomeBand, Occupation, Education, Gender, MaritalStatus, Age, TotalChildren, NumberCarsOwned, HomeOwnerFlag | mixed | individual demographics (empty for stores) |
| Segment | text | `Individual: <income band>` or `Store: <business type>` |
| cm_rank, abc_class_cm | int/text | customer CM rank and ABC class |
| churn_prob | dec | probability of no order in the next 6 months (LightGBM, scored at end of data) |
| churn_risk_band | text | `Low (<30%)`, `Medium (30-60%)`, `High (60-80%)`, `Very high (>80%)` |

## pareto_abc.csv — ABC table for every dimension, stacked (long format)

| Column | Description |
|---|---|
| dimension | `customer`, `customer_type`, `channel`, `segment`, `store_business_type`, `store_size`, `income_band`, `occupation`, `rfm_segment`, `product`, `subcategory`, `category`, `territory`, `country`, `channel_category` |
| entity_key, entity_name, parent | identifier, display name and parent level (e.g. category for products, territory group for territories, channel for `channel_category`) |
| revenue, cogs, cm, cm_margin, lines | totals |
| rank, entity_share, cm_share, cum_cm_share, cum_positive_cm_share, abc_class | as in dim_product |

## pareto_concentration.csv — share of CM from the top x% (dimensions with ≥ 20 entities)

`dimension`, `top_share` (0.01–0.5), `top_n`, `revenue_share`, `cm_share_of_net`, `cm_share_of_positive`.

## abc_class_summary.csv — per dimension and class

`dimension`, `abc_class`, `entities`, `revenue`, `cm`, `entity_share`, `revenue_share`, `cm_share_of_net`.

## cm_summary.csv — CM1 vs CM2 by channel (+ Total)

`Channel`, `revenue`, `cogs`, `freight_alloc`, `cm1`, `cm2`, `cm1_margin`, `cm2_margin`.

## distance_comparison.csv — candidate "distant" definitions per territory

`TerritoryGroup`, `Territory`, `median_distance_km`, `revenue`, `freight_pct_of_revenue`, and the revenue share
flagged as distant by each option: `share_distant_territory_group`, `share_distant_great_circle`,
`share_distant_freight_per_unit` (orders in the top quartile of freight per unit).

## elasticity.csv — one row per channel × category × territory group

| Column | Description |
|---|---|
| Channel, Category, TerritoryGroup | cell |
| elasticity_ols, se_ols, p_ols | fixed-effects OLS estimate, cluster-robust SE, p-value |
| n_cells | panel cells with price variation in this cell |
| elasticity_lgbm | LightGBM arc elasticity (±5% price) |
| plausible | estimate significant, in plausible range and n_cells ≥ min_obs |
| elasticity_estimated, estimated_source | value used by the "estimated" scenario and where it comes from |
| elasticity_low, elasticity_mid, elasticity_high | assumed values (−0.5, −1.5, −2.5) |

## elasticity_price_variation.csv — identification check

`Channel`, `cells`, `sd_within_log_price` (std. dev. of log price within product × territory), `share_cells_with_price_change`.

## churn_metrics.csv — out-of-time test metrics

`model` (`logistic_regression`, `lightgbm`), `segment` (`All`, `Individual`, `Store`), `roc_auc`, `pr_auc`, `brier`,
`precision`, `recall`, `f1` (threshold 0.5), `base_rate`, `top_decile_churn_rate`.

## churn_feature_importance.csv / churn_lr_coefficients.csv

Permutation importance of the LightGBM model on the test set (`permutation_importance_auc` = ROC-AUC drop,
`permutation_std`); logistic regression coefficients on standardised / one-hot features.

## scenario_results.csv — one row per scenario × segment (main Power BI scenario table)

| Column | Description |
|---|---|
| scenario_id | `<markup_type>\|<low_cm_rule>\|<elasticity_scenario>\|<markup_level>` |
| markup_type | `low_cm`, `distant`, `both` |
| low_cm_rule | `abc_c`, `margin_below`, `n/a` (distant only) |
| elasticity_scenario | `estimated`, `low`, `mid`, `high` |
| markup_level | 0.00–0.20 |
| dimension | `Total`, `Channel`, `CustomerType`, `TerritoryGroup`, `Territory`, `Category` |
| dimension_value | segment value (`Total` for the total row) |
| revenue_base, revenue | baseline-year revenue before / after markup (incl. spillover) |
| units_base, units | units before / after |
| cm_base, cm | CM before / after |
| spillover_cm_lost | CM lost because lost customers stop buying non-marked-up products (negative = gain) |
| affected_revenue_share | share of baseline revenue that is marked up |
| customers_base | customers in the baseline year (customer dimensions only; empty for Category) |
| expected_active_base | Σ(1 − baseline churn probability) |
| customers_lost | expected additional customers lost due to the markup |
| cm_change, cm_change_pct | cm − cm_base; relative to abs(cm_base) |
| revenue_change_pct, volume_change_pct | relative changes |
| customers_lost_pct | customers_lost / expected_active_base |
| customer_loss_unacceptable | customers_lost_pct > `max_customer_loss_pct` (5%) |

## scenario_summary.csv — the `Total` rows of scenario_results (one row per scenario, 220)

## scenario_optimum.csv — per markup type × low-CM rule × elasticity scenario

`cm_baseline`, `profit_max_level` / `profit_max_cm` / `profit_max_cm_change` / `profit_max_customers_lost_pct`
(unconstrained optimum within the grid), `first_unacceptable_level` (first level above the loss limit, empty if none),
`recommended_level` / `recommended_cm_change` / `recommended_customers_lost_pct` (best level within the loss limit).

## cleaning_log.csv

`check`, `table`, `rows_affected`, `action` — every data-quality check run by `src/cleaning.py`.

## run_summary.json

Run metadata: freight/distance correlations, OLS summary, churn sample sizes and base rates, row count per file.

---

# Input contract (`data/raw/`)

To run the pipeline on other (e.g. re-cleaned) data, provide these CSVs with at least these columns, then
*Run pipeline (from data/raw)*:

| File | Required columns |
|---|---|
| sales_order_header.csv | SalesOrderID, OrderDate, DueDate, ShipDate, Status, OnlineOrderFlag, CustomerID, SalesPersonID, TerritoryID, ShipToAddressID, ShipMethodID, SubTotal, TaxAmt, Freight, TotalDue |
| sales_order_detail.csv | SalesOrderID, SalesOrderDetailID, ProductID, SpecialOfferID, OrderQty, UnitPrice, UnitPriceDiscount, LineTotal |
| product.csv | ProductID, Name, ProductNumber, ProductSubcategoryID, StandardCost, ListPrice, SellStartDate, SellEndDate, DiscontinuedDate (+ optional attributes) |
| product_subcategory.csv | ProductSubcategoryID, ProductCategoryID, Name |
| product_category.csv | ProductCategoryID, Name |
| product_cost_history.csv | ProductID, StartDate, EndDate, StandardCost |
| product_list_price_history.csv | ProductID, StartDate, EndDate, ListPrice |
| customer.csv | CustomerID, PersonID, StoreID, TerritoryID |
| store_demographics.csv | StoreID, StoreName, AnnualSales, BusinessType, Specialty, NumberEmployees (+ optional survey fields) |
| person_demographics.csv | PersonID, BirthDate, MaritalStatus, YearlyIncome, Gender, TotalChildren, Education, Occupation, HomeOwnerFlag, NumberCarsOwned |
| sales_territory.csv | TerritoryID, Name, CountryRegionCode, TerritoryGroup |
| address.csv | AddressID, City, StateProvinceID, PostalCode, Latitude, Longitude |
| state_province.csv | StateProvinceID, StateProvinceCode, Name, CountryRegionCode, TerritoryID |
| special_offer.csv | SpecialOfferID, Description, DiscountPct, Type, Category, StartDate, EndDate, MinQty, MaxQty |
| ship_method.csv | ShipMethodID, Name, ShipBase, ShipRate |
