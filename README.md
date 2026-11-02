# AdventureWorks – Contribution Margin & Markup Simulation

Group 3 · Business Reports and Dashboards (FH Joanneum).
Contribution margin model, Pareto/ABC analysis, churn and price-sensitivity models and a markup scenario engine on
AdventureWorks2022 (OLTP), with a Streamlit dashboard and CSV outputs for Power BI.

## Layout

```
docker/              Docker setup (run everything from here)
  manage.sh            menu: build / start / stop / pipeline / notebook / URLs / shell
  docker-compose.yml   db (SQL Server 2022 + AdventureWorks), app (Python + JupyterLab), dashboard (Streamlit)
  Dockerfile           Python image used by app and dashboard
  mssql_entrypoint.sh  restores the backup on first DB start
  .env.example         copy to .env and set the SA password
config.yaml          all settings (paths, freight on/off, ABC thresholds, distance, churn window, markup grid)
src/                 pipeline modules
  extract.py           SQL Server → data/raw/*.csv
  cleaning.py          validation + cleaning log → data/processed/
  cm.py                order-line CM fact table (CM1 / CM2)
  segments.py          customer table, derived segments, RFM
  pareto.py            Pareto / ABC
  distance.py          great-circle distance, distant flag, option comparison
  elasticity.py        FE-OLS + LightGBM elasticities, assumed fallbacks
  churn.py             churn features, logistic regression + LightGBM, scoring
  scenarios.py         markup scenario engine + grid + optimum
  export.py            CSV exports
  pipeline.py          runs everything: python -m src.pipeline [--extract]
dashboard/app.py     Streamlit dashboard (live scenario engine)
notebooks/analysis.ipynb  end-to-end walkthrough with presentation-ready text
data/source/         AdventureWorks2022.bak (downloaded by manage.sh)
data/raw/            extracted tables (input contract in SCHEMA.md)
data/processed/      cleaned tables and model artefacts
outputs/             CSVs for Power BI + figures/
ASSUMPTIONS.md       every modelling assumption (for the presentation)
SCHEMA.md            schema of every output file + Power BI import notes
```

## Getting started

```bash
cd docker
./manage.sh
#  3  Build Containers      (downloads the backup if missing, starts db + app + dashboard)
# 11  Run pipeline (re-extract data/raw from SQL Server)   – first run
# 10  Run pipeline (from data/raw)                         – after changing config.yaml
# 12  Execute analysis notebook
```

- Dashboard: http://localhost:8501
- JupyterLab: http://localhost:8888
- SQL Server: `localhost,1433`, database `AdventureWorks2022`, user `sa`, password in `docker/.env`

The CSVs in `outputs/` are the hand-over for Power BI (see `SCHEMA.md` for the schema and the locale setting needed
on German/Austrian Windows).
