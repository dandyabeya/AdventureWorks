#!/bin/bash
# -----------------------------------------------------------------------------
# mssql_entrypoint.sh – Start SQL Server and restore AdventureWorks on first run
# -----------------------------------------------------------------------------
set -e

SQLCMD=/opt/mssql-tools18/bin/sqlcmd
MARKER=/var/opt/mssql/.restored
BAK="/backup/${MSSQL_DB}.bak"

/opt/mssql/bin/sqlservr &
PID=$!

until $SQLCMD -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C -Q "SELECT 1" >/dev/null 2>&1; do
    sleep 2
done

if [ ! -f "$MARKER" ]; then
    if [ ! -f "$BAK" ]; then
        echo "❌ Backup not found: $BAK (run manage.sh → 'Download AdventureWorks backup')"
        kill $PID; exit 1
    fi
    echo "Restoring ${MSSQL_DB} from ${BAK} ..."
    $SQLCMD -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C -b -Q "
        RESTORE DATABASE [${MSSQL_DB}] FROM DISK = '${BAK}'
        WITH MOVE '${MSSQL_DB}'     TO '/var/opt/mssql/data/${MSSQL_DB}.mdf',
             MOVE '${MSSQL_DB}_log' TO '/var/opt/mssql/data/${MSSQL_DB}_log.ldf',
             REPLACE"
    touch "$MARKER"
    echo "✅ Restore complete."
fi

wait $PID
