import os

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, command, Output
from azure.ai.ml.entities import Model

# -------------------------------------------------------------------
# Workspace config (taken from your setup)
# You can also override via environment variables in GitHub Actions.
# -------------------------------------------------------------------
SUBSCRIPTION_ID = os.getenv(
    "AZURE_SUBSCRIPTION_ID", "a485bb50-61aa-4b2f-bc7f-b6b53539b9d3"
)
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "rg-60104832")
WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME", "ml-lab5-mri")

# Compute to run the job on (must exist in your workspace)
# From your screenshots, your compute name is also "lab5-mri".
COMPUTE_NAME = os.getenv("AML_COMPUTE_NAME", "lab5-mri")


def main() -> None:
    # Connect to Azure ML
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )

    # ------------------------------------------------------------
    # 1. Define a command job that runs train_sklearn.py on Azure ML
    # ------------------------------------------------------------
    train_job = command(
        display_name="train_tumor_model",
        description="Train a simple RandomForest for tumor detection (synthetic data).",
        code="./pipeline",  # folder that contains train_sklearn.py
        command=(
            "python train_sklearn.py "
            "--output_dir ${{outputs.model_output}}"
        ),
        environment="AzureML-sklearn-1.3-ubuntu20.04-py38-cpu",
        compute=COMPUTE_NAME,
        experiment_name="tumor_mri_training",
        outputs={
            "model_output": Output(type="uri_folder", mode="rw_mount"),
        },
    )

    submitted_job = ml_client.jobs.create_or_update(train_job)
    print(f"Submitted job: {submitted_job.name}")

    # Stream logs until the job finishes
    ml_client.jobs.stream(submitted_job.name)

    # Refresh job to get final output URIs
    completed_job = ml_client.jobs.get(submitted_job.name)
    model_folder_uri = completed_job.outputs["model_output"].uri

    print(f"Model folder URI: {model_folder_uri}")

    # ------------------------------------------------------------
    # 2. Register the model in Azure ML
    # ------------------------------------------------------------
    model = Model(
        name="tumor_detection_model",
        path=model_folder_uri,
        type="custom_model",
        description="RandomForest tumor detection model trained via pipeline_job.py",
    )

    registered_model = ml_client.models.create_or_update(model)
    print(
        f"Model registered: {registered_model.name}, "
        f"version: {registered_model.version}"
    )


if __name__ == "__main__":
    main()
