import sys
from pathlib import Path

from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlops.mlflow_utils import promote_model


MODEL_NAME = "PortfolioXGBoost"
TARGET_STAGE = "Production"


def latest_registered_version(model_name: str) -> int:
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError(f"No versions found for {model_name}")
    return max(int(version.version) for version in versions)


def main():
    version = latest_registered_version(MODEL_NAME)
    promote_model(MODEL_NAME, version, TARGET_STAGE)
    print(f"Promoted {MODEL_NAME} version {version} to {TARGET_STAGE}")


if __name__ == "__main__":
    main()