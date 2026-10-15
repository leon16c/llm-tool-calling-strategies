from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PACKAGE_ROOT.parent

TOOLS_SCHEMA_PATH = PACKAGE_ROOT / "schemas" / "tools.json"
RUN_LOG_SCHEMA_PATH = PACKAGE_ROOT / "schemas" / "run_log.schema.json"
REQUEST_ORACLE_PATH = REPOSITORY_ROOT / "evaluation" / "requests" / "initial_requests.json"
REQUEST_ORACLE_SCHEMA_PATH = (
    REPOSITORY_ROOT / "evaluation" / "requests" / "initial_requests.schema.json"
)
