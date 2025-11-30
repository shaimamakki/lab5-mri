import os

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment

# Workspace config
SUBSCRIPTION_ID = "a485bb50-61aa-4b2f-bc7f-b6b53539b9d3"
RESOURCE_GROUP = "rg-60104832"
WORKSPACE_NAME = "lab5-mri"

ENDPOINT_NAME = "tumor-endpoint-60104832"
DEPLOYMENT_NAME = "blue"


def main() -> None:

    # Connect to workspace
    ml_client = MLClient(
        DefaultAzureCredential(),
        SUBSCRIPTION_ID,
        RESOURCE_GROUP,
        WORKSPACE_NAME,
    )

    # ------------------------------------------------------------
    # 1. Create or get the endpoint
    # ------------------------------------------------------------
    try:
        endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
        print(f"Using existing endpoint: {endpoint.name}")
    except Exception:
        print(f"Creating new endpoint: {ENDPOINT_NAME}")
        endpoint = ManagedOnlineEndpoint(
            name=ENDPOINT_NAME,
            auth_mode="key",
            description="Tumor MRI classifier endpoint",
        )
        endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        print(f"Endpoint created: {endpoint.name}")

    # ------------------------------------------------------------
    # 2. Get the latest registered model
    # ------------------------------------------------------------
    models = list(ml_client.models.list(name="tumor_detection_model"))
    if not models:
        raise RuntimeError("No model named 'tumor_detection_model' found.")

    latest_model = sorted(models, key=lambda m: int(m.version))[-1]
    print(f"Using model: {latest_model.name} v{latest_model.version}")

    # ------------------------------------------------------------
    # 3. Deployment config
    # ------------------------------------------------------------
    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=latest_model,
        code_path="./pipeline",        # << your score.py is here
        scoring_script="score.py",
        environment="AzureML-sklearn-1.0-ubuntu20.04-py38",
        instance_type="Standard_DS2_v2",
        instance_count=1,
    )

    ml_client.online_deployments.begin_create_or_update(deployment).result()

    # Route 100% traffic to deployment
    ml_client.online_endpoints.begin_traffic_update(
        name=ENDPOINT_NAME,
        traffic={DEPLOYMENT_NAME: 100}
    ).result()

    print(f"Deployment ready! Endpoint: {ENDPOINT_NAME}")


if __name__ == "__main__":
    main()
