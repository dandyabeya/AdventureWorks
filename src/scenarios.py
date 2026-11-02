"""Markup scenario engine.

For each order line in the baseline year, a markup multiplies the price of the
affected lines; volume reacts through the elasticity; churn rises for customers
in proportion to the share of their spend that is marked up; and customers who
are lost also stop buying the products that were *not* marked up (spillover).

    mult       = (1 + m)^[low-CM line] * (1 + m)^[distant line]
    q_factor   = mult ^ elasticity(channel, category, territory group)
    revenue'   = revenue * mult * q_factor
    cogs'      = cogs * q_factor
    freight'   = freight * mult * q_factor      (freight is a % of order value)
    v_c        = sum_affected revenue*(1 - q_factor) / sum revenue      (customer c)
    dp_c       = (1 - p0_c) * churn_link * v_c                          (churn uplift)
    spillover  = dp_c * CM' of c's non-affected lines
"""
import numpy as np
import pandas as pd

from . import elasticity as el

SEGMENT_DIMS = ["Channel", "CustomerType", "TerritoryGroup", "Territory", "Category"]
CUSTOMER_DIMS = ["Channel", "CustomerType", "TerritoryGroup", "Territory"]  # dims a customer belongs to exactly once


def low_cm_products(product_abc, cfg):
    """{rule: set of ProductIDs} for every low-CM definition."""
    sc = cfg["scenarios"]
    return {
        "abc_c": set(product_abc.loc[product_abc.abc_class.isin(["C", "Negative"]), "ProductID"]),
        "margin_below": set(product_abc.loc[product_abc.cm_margin < sc["low_cm_margin_threshold"], "ProductID"]),
    }


def baseline_lines(fact, months):
    """Last ``months`` months of order lines per channel (annual baseline)."""
    end = fact.groupby("Channel").OrderDate.transform("max")
    return fact[fact.OrderDate > end - pd.DateOffset(months=months)].copy()


class ScenarioEngine:
    def __init__(self, fact, product_abc, elasticity_table, churn_scores, cfg):
        self.cfg = cfg
        sc = cfg["scenarios"]
        b = baseline_lines(fact, sc["baseline_months"])
        b = b.merge(churn_scores, on="CustomerID", how="left")
        b["churn_prob"] = b.churn_prob.fillna(b.churn_prob.median())
        self.lines = b.reset_index(drop=True)
        self.low_cm = {rule: b.ProductID.isin(ids).values for rule, ids in low_cm_products(product_abc, cfg).items()}
        self.distant = b.is_distant.values
        self.elasticity_table = elasticity_table
        self.cust_codes, self.cust_index = pd.factorize(b.CustomerID)
        p0 = b.groupby("CustomerID").churn_prob.first()
        self.p0 = p0.reindex(self.cust_index).values
        self.freight_in_cm = cfg["cm"]["freight_allocation"]

    def elasticities(self, scenario, override=None):
        """Per-line elasticity for a scenario name, or a single ``override`` value for all lines."""
        if override is not None:
            return np.full(len(self.lines), float(override))
        e = el.lookup(self.elasticity_table, scenario)
        keys = pd.MultiIndex.from_frame(self.lines[["Channel", "Category", "TerritoryGroup"]])
        fallback = self.cfg["elasticity"]["assumed"]["mid" if scenario == "estimated" else scenario]
        return e.reindex(keys).fillna(fallback).values

    def simulate(self, markup_type, level, low_cm_rule="abc_c", elasticity_scenario="mid",
                 churn_link=None, include_spillover=None, elasticity_override=None):
        """Line-level results for one scenario (DataFrame aligned with self.lines)."""
        sc = self.cfg["scenarios"]
        churn_link = sc["churn_link"] if churn_link is None else churn_link
        include_spillover = sc["include_spillover"] if include_spillover is None else include_spillover
        b = self.lines
        low = self.low_cm[low_cm_rule] if markup_type in ("low_cm", "both") else np.zeros(len(b), bool)
        dist = self.distant if markup_type in ("distant", "both") else np.zeros(len(b), bool)
        mult = (1 + level) ** low.astype(int) * (1 + level) ** dist.astype(int)
        q = mult ** self.elasticities(elasticity_scenario, elasticity_override)
        affected = low | dist

        revenue = b.revenue.values * mult * q
        cogs = b.cogs.values * q
        freight = b.freight_alloc.values * mult * q
        cm = revenue - cogs - (freight if self.freight_in_cm else 0)

        # churn uplift per customer from the revenue-weighted volume loss on affected lines
        lost_rev = np.bincount(self.cust_codes, weights=b.revenue.values * (1 - q) * affected)
        base_rev = np.bincount(self.cust_codes, weights=b.revenue.values)
        v = np.divide(lost_rev, base_rev, out=np.zeros_like(lost_rev), where=base_rev > 0)
        dp = (1 - self.p0) * churn_link * np.clip(v, 0, 1)
        dp_line = dp[self.cust_codes]
        spill = dp_line * (~affected) if include_spillover else np.zeros(len(b))

        return pd.DataFrame({
            "revenue_base": b.revenue.values, "revenue": revenue - spill * revenue,
            "units_base": b.OrderQty.values, "units": b.OrderQty.values * q * (1 - spill),
            "cm_base": b.cm.values, "cm": cm - spill * cm, "spillover_cm_lost": spill * cm,
            "affected": affected, "price_multiplier": mult,
        }), dp

    def totals(self, res, dp):
        """Headline totals for one simulate() result (fast path for curves)."""
        expected_active = (1 - self.p0).sum()
        return {"revenue_base": res.revenue_base.sum(), "revenue": res.revenue.sum(),
                "units_base": res.units_base.sum(), "units": res.units.sum(),
                "cm_base": res.cm_base.sum(), "cm": res.cm.sum(), "spillover_cm_lost": res.spillover_cm_lost.sum(),
                "customers_base": len(self.p0), "expected_active_base": expected_active,
                "customers_lost": dp.sum(), "customers_lost_pct": dp.sum() / expected_active}

    def aggregate(self, res, dp, keys=None):
        """Aggregate simulate() output to Total and every segment dimension."""
        keys = keys or {}
        b = self.lines
        out = []
        groups = {"Total": pd.Series("Total", index=b.index)}
        groups.update({d: b[d] for d in SEGMENT_DIMS})
        cust = pd.DataFrame({"CustomerID": self.cust_index, "dp": dp, "p0": self.p0})
        cust_attr = b.groupby("CustomerID")[CUSTOMER_DIMS].first()
        cust = cust.join(cust_attr, on="CustomerID")
        cust["Total"] = "Total"
        for dim, labels in groups.items():
            g = res.groupby(labels.values).agg(
                revenue_base=("revenue_base", "sum"), revenue=("revenue", "sum"),
                units_base=("units_base", "sum"), units=("units", "sum"),
                cm_base=("cm_base", "sum"), cm=("cm", "sum"), spillover_cm_lost=("spillover_cm_lost", "sum"),
                affected_revenue_share=("affected", lambda s: np.average(s, weights=res.loc[s.index, "revenue_base"])),
            )
            if dim in CUSTOMER_DIMS + ["Total"]:
                c = cust.groupby(dim).agg(customers_base=("CustomerID", "size"),
                                          expected_active_base=("p0", lambda s: (1 - s).sum()),
                                          customers_lost=("dp", "sum"))
                g = g.join(c)
            g = g.reset_index(names="dimension_value")
            g.insert(0, "dimension", dim)
            out.append(g)
        df = pd.concat(out, ignore_index=True)
        for k, v in reversed(list(keys.items())):
            df.insert(0, k, v)
        return df

    def run_grid(self):
        """Run every configured scenario. Returns (segment-level results, total-level results)."""
        sc = self.cfg["scenarios"]
        frames = []
        for mtype in sc["markup_types"]:
            rules = sc["low_cm_rules"] if mtype in ("low_cm", "both") else ["n/a"]
            for rule in rules:
                for eps in sc["elasticity_scenarios"]:
                    for level in sc["markup_levels"]:
                        res, dp = self.simulate(mtype, level, rule if rule != "n/a" else "abc_c", eps)
                        keys = {"scenario_id": f"{mtype}|{rule}|{eps}|{level:.2f}", "markup_type": mtype,
                                "low_cm_rule": rule, "elasticity_scenario": eps, "markup_level": level}
                        frames.append(self.aggregate(res, dp, keys))
        seg = pd.concat(frames, ignore_index=True)
        seg["cm_change"] = seg.cm - seg.cm_base
        seg["cm_change_pct"] = seg.cm_change / seg.cm_base.abs()
        seg["revenue_change_pct"] = seg.revenue / seg.revenue_base - 1
        seg["volume_change_pct"] = seg.units / seg.units_base - 1
        seg["customers_lost_pct"] = seg.customers_lost / seg.expected_active_base
        seg["customer_loss_unacceptable"] = seg.customers_lost_pct > sc["max_customer_loss_pct"]
        total = seg[seg.dimension == "Total"].drop(columns=["dimension", "dimension_value"]).reset_index(drop=True)
        return seg, total


def optimum(total, cfg):
    """Per markup type x rule x elasticity: profit-maximising level, with and without the customer-loss limit."""
    limit = cfg["scenarios"]["max_customer_loss_pct"]
    rows = []
    for (mtype, rule, eps), g in total.groupby(["markup_type", "low_cm_rule", "elasticity_scenario"]):
        g = g.sort_values("markup_level")
        best = g.loc[g.cm.idxmax()]
        ok = g[g.customers_lost_pct <= limit]
        best_ok = ok.loc[ok.cm.idxmax()] if len(ok) else None
        breach = g[g.customers_lost_pct > limit]
        rows.append({
            "markup_type": mtype, "low_cm_rule": rule, "elasticity_scenario": eps,
            "cm_baseline": g.cm_base.iloc[0],
            "profit_max_level": best.markup_level, "profit_max_cm": best.cm, "profit_max_cm_change": best.cm_change,
            "profit_max_customers_lost_pct": best.customers_lost_pct,
            "first_unacceptable_level": breach.markup_level.min() if len(breach) else np.nan,
            "recommended_level": best_ok.markup_level if best_ok is not None else 0.0,
            "recommended_cm_change": best_ok.cm_change if best_ok is not None else 0.0,
            "recommended_customers_lost_pct": best_ok.customers_lost_pct if best_ok is not None else 0.0,
        })
    return pd.DataFrame(rows)
