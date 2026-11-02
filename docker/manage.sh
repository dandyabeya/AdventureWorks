#!/bin/bash
# -----------------------------------------------------------------------------
# manage.sh – Helper script for managing the AdventureWorks Docker environment
# -----------------------------------------------------------------------------

set -e

# --- Verify directory --------------------------------------------------------
CURRENT_DIR=$(basename "$(pwd)")
if [ "$CURRENT_DIR" != "docker" ]; then
    echo "❌ This script must be run from the 'docker/' directory."
    echo "➡️  cd /opt/github_proj/projects/AdventureWorks/docker"
    exit 1
fi

# --- Variables ---------------------------------------------------------------
COMPOSE="docker compose"
DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DOCKER_DIR/.." && pwd)"
ENV_FILE="${DOCKER_DIR}/.env"
ENV_EXAMPLE="${DOCKER_DIR}/.env.example"
COMPOSE_FILE="${DOCKER_DIR}/docker-compose.yml"

BAK_URL="https://github.com/Microsoft/sql-server-samples/releases/download/adventureworks/AdventureWorks2022.bak"
BAK_SHA256="d17567adb1521f972e1dc183a7216cea869c4580b5d75632425bcadbaf82ce5e"
BAK_FILE="${REPO_ROOT}/data/source/AdventureWorks2022.bak"

if [ ! -f "$ENV_FILE" ]; then
    if [ -f "$ENV_EXAMPLE" ]; then
        cp "$ENV_EXAMPLE" "$ENV_FILE"
        echo "[init] Created docker/.env from docker/.env.example — set MSSQL_SA_PASSWORD before building."
    else
        echo "[warn] No .env or .env.example in docker/. Environment variables may be missing."
    fi
fi

if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

COMPOSE_PROJECT_NAME=${COMPOSE_PROJECT_NAME:-adventureworks}

# Run the app container as the host user so files written to the repo stay ours
export HOST_UID HOST_GID
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

# --- Helpers -----------------------------------------------------------------

compose_in() {
    $COMPOSE --env-file "$ENV_FILE" -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" "$@"
}

env_val() {
    local key="$1" default="$2"
    local val
    val="$(grep -E "^\s*${key}\s*=" "$ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"' | tr -d "'" | xargs)"
    echo "${val:-$default}"
}

ensure_backup() {
    if [ -f "$BAK_FILE" ]; then
        return 0
    fi
    download_backup
}

# --- Functions ---------------------------------------------------------------

show_all_containers() {
    echo "📦 Showing ALL Docker containers (running + stopped):"
    docker ps -a --format "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}"
}

download_backup() {
    mkdir -p "$(dirname "$BAK_FILE")"
    if [ -f "$BAK_FILE" ]; then
        echo "ℹ️  Backup already present: $BAK_FILE"
    else
        echo "⬇️  Downloading AdventureWorks2022.bak (~210 MB)..."
        curl -fSL -o "$BAK_FILE.part" "$BAK_URL"
        mv "$BAK_FILE.part" "$BAK_FILE"
    fi
    echo "🔐 Verifying checksum..."
    if echo "${BAK_SHA256}  ${BAK_FILE}" | sha256sum -c --status; then
        echo "✅ Backup OK: $BAK_FILE"
    else
        echo "❌ Checksum mismatch — delete $BAK_FILE and download again."
        return 1
    fi
}

clean_build_containers() {
    ensure_backup
    echo "🏗️  Building images without cache (db, app, dashboard)..."
    compose_in down --remove-orphans
    compose_in build --no-cache --force-rm
    compose_in up -d
    echo ""
    echo "✅ Clean build finished!"
    print_urls
}

build_containers() {
    ensure_backup
    echo "🏗️  Building images with cache (db, app, dashboard)..."
    compose_in build
    compose_in up -d
    echo ""
    echo "✅ Build finished!"
    print_urls
}

start_containers() {
    ensure_backup
    echo "▶️  Starting containers (first start restores the database, ~30s)..."
    compose_in up -d
    print_urls
}

stop_containers() {
    echo "⏹️  Stopping containers..."
    compose_in stop
    echo "✅ Containers stopped (data kept)."
}

remove_containers() {
    echo "🧹 Stopping and removing project containers and volumes..."
    echo "   (the database is restored again from the backup on next start)"
    compose_in down --volumes --remove-orphans
    echo ""
    echo "✅ Project containers and volumes removed."
}

clean_orphans() {
    echo "🧽 Cleaning up orphaned containers..."
    local orphans
    orphans="$(docker ps -aq --filter "name=${COMPOSE_PROJECT_NAME}-.*-run-" 2>/dev/null)"
    if [ -n "$orphans" ]; then
        docker rm -f $orphans
        echo "✅ Removed orphan containers:"
        echo "$orphans"
    else
        echo "ℹ️  No orphan containers found."
    fi
}

run_pipeline() {
    echo "⚙️  Running pipeline (clean → CM → ABC → elasticity → churn → scenarios → CSV)..."
    compose_in exec -T app python -m src.pipeline "$@"
    echo "🔄 Restarting dashboard to load the new outputs..."
    compose_in restart dashboard
    echo "✅ Outputs in outputs/ (CSV for Power BI, see SCHEMA.md)."
}

run_notebook() {
    echo "📓 Executing notebooks/analysis.ipynb in place..."
    compose_in exec -T app jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=1800 notebooks/analysis.ipynb
    echo "✅ Notebook executed."
}

app_shell() {
    echo "🐚 Opening shell in ${COMPOSE_PROJECT_NAME}_app (type 'exit' to leave)..."
    compose_in exec app bash
}

print_urls() {
    local jupyter_port db_port db_name
    jupyter_port="$(env_val JUPYTER_PORT 8888)"
    db_port="$(env_val DB_PORT 1433)"
    db_name="$(env_val MSSQL_DB AdventureWorks2022)"
    echo ""
    echo "------------------------------------------------------------"
    echo " Service URLs"
    echo "------------------------------------------------------------"
    echo "  Dashboard        : http://localhost:$(env_val DASHBOARD_PORT 8501)"
    echo "  JupyterLab       : http://localhost:${jupyter_port}"
    echo "  SQL Server       : localhost,${db_port}  (db: ${db_name}, user: sa)"
    echo "  Password         : MSSQL_SA_PASSWORD in docker/.env"
    echo "------------------------------------------------------------"
}

# --- Menu --------------------------------------------------------------------

show_menu() {
    echo ""
    echo "------------------------------------------------------------"
    echo " 🧩 AdventureWorks Docker Management"
    echo "------------------------------------------------------------"
    echo " Project:   $(basename "$REPO_ROOT")"
    echo " Directory: $DOCKER_DIR"
    echo " Compose:   ${COMPOSE_PROJECT_NAME}"
    echo "------------------------------------------------------------"
    echo "  1) Show all Docker containers (global)"
    echo "  2) Clean Build Containers - no cache (db, app, dashboard)"
    echo "  3) Build Containers - with cache (db, app, dashboard)"
    echo "  4) Start Containers"
    echo "  5) Stop  Containers"
    echo "  6) Remove Containers + volumes (db, app, dashboard)"
    echo "  7) Clean Orphan Containers"
    echo "  8) Print service URLs"
    echo "  9) Download AdventureWorks backup"
    echo " 10) Run pipeline (from data/raw)"
    echo " 11) Run pipeline (re-extract data/raw from SQL Server)"
    echo " 12) Execute analysis notebook"
    echo " 13) Open shell in app container"
    echo " 14) Exit"
    echo "------------------------------------------------------------"
}

# --- Menu Loop ---------------------------------------------------------------
while true; do
    show_menu
    echo ""
    read -rp "Select an action (1-14): " choice
    echo ""
    case "$choice" in
        1)  show_all_containers ;;
        2)  clean_build_containers ;;
        3)  build_containers ;;
        4)  start_containers ;;
        5)  stop_containers ;;
        6)  remove_containers ;;
        7)  clean_orphans ;;
        8)  print_urls ;;
        9)  download_backup ;;
        10) run_pipeline ;;
        11) run_pipeline --extract ;;
        12) run_notebook ;;
        13) app_shell ;;
        14) echo "👋 Exiting."; exit 0 ;;
        *)  echo "❓ Invalid choice, please enter 1-14." ;;
    esac
    echo ""
    read -rp "Press Enter to continue (output will remain visible)..."
    echo ""
done
