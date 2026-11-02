"""Run the full pipeline end to end.

    python -m src.pipeline              # from data/raw CSVs
    python -m src.pipeline --extract    # first re-extract data/raw from SQL Server
    python -m src.pipeline --config other_config.yaml
"""
import argparse
import json
import time

import pandas as pd

from . import churn, cleaning, cm, distance, elasticity, export, pareto, scenarios, segments
from .config import load_config


def run(config_path=None, extract=False, verbose=True):
    cfg = load_config(config_path)
    paths = cfg["paths"]
    t0 = time.time()

    def step(msg):
        if verbose:
            print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    if extract:
        from .extract import extract_all
        step("Extracting tables from SQL Server")
        extract_all(paths["raw_dir"])

    step("Cleaning raw tables")
    tables, log = cleaning.clean(cleaning.load_raw(paths["raw_dir"]))
    cleaning.save_processed(tables, log, paths["processed_dir"])

    step("Building CM fact table")
    fact = cm.build_fact(tables, cfg)
    customers = segments.build_customers(fact, tables)

    step("Pareto / ABC analysis")
    pareto_results = pareto.run_all(fact, customers, cfg)
    product_abc = pareto_results["product"]
    concentration = pd.concat([pareto.concentration(g).assign(dimension=d) for d, g in pareto_results.items()
                               if len(g) >= 20], ignore_index=True)
    class_summary = pd.concat([pareto.class_summary(g).assign(dimension=d) for d, g in pareto_results.items()],
                              ignore_index=True)
    dist_cmp, dist_corr = distance.compare_definitions(fact, cfg)

    step("Estimating price elasticities")
    elasticity_table, el_diag = elasticity.estimate(fact, cfg)

    step("Training churn models")
    churn_result = churn.train_and_evaluate(fact, customers, cfg)

    step("Running scenario grid")
    engine = scenarios.ScenarioEngine(fact, product_abc, elasticity_table, churn_result["scored"], cfg)
    seg, total = engine.run_grid()
    opt = scenarios.optimum(total, cfg)

    step("Writing outputs")
    # artefacts the dashboard and notebook reuse
    fact.to_parquet(paths["processed_dir"] / "fact.parquet", index=False)
    customers.to_parquet(paths["processed_dir"] / "customers.parquet", index=False)
    product_abc.to_parquet(paths["processed_dir"] / "product_abc.parquet", index=False)
    elasticity_table.to_parquet(paths["processed_dir"] / "elasticity.parquet", index=False)
    churn_result["scored"].to_parquet(paths["processed_dir"] / "churn_scores.parquet", index=False)

    files = export.write_all(
        paths["output_dir"], fact, customers, pareto_results, product_abc,
        scenarios.low_cm_products(product_abc, cfg), churn_result, elasticity_table, seg, total, opt,
        extras={"concentration": concentration, "class_summary": class_summary,
                "cm_summary": cm.cm_summary(fact), "distance_comparison": dist_cmp,
                "within_variation": el_diag["within_variation"], "cleaning_log": log})
    summary = {"distance_freight_correlation": dist_corr, "elasticity_ols": el_diag["ols"],
               "churn_best_model": churn_result["best_model"], "churn_train_size": churn_result["train_size"],
               "churn_test_size": churn_result["test_size"], "churn_train_base_rate": churn_result["train_base_rate"],
               "churn_test_base_rate": churn_result["test_base_rate"], "files": files}
    with open(paths["output_dir"] / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    step(f"Done — {len(files)} files in {paths['output_dir']}")
    return {"cfg": cfg, "tables": tables, "fact": fact, "customers": customers, "pareto": pareto_results,
            "elasticity": elasticity_table, "elasticity_diagnostics": el_diag, "churn": churn_result,
            "engine": engine, "scenario_segments": seg, "scenario_total": total, "scenario_optimum": opt,
            "distance_comparison": dist_cmp, "distance_correlation": dist_corr, "cleaning_log": log,
            "summary": summary}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--extract", action="store_true", help="re-extract data/raw from SQL Server first")
    args = ap.parse_args()
    run(args.config, args.extract)


if __name__ == "__main__":
    main()
