from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import (
    FeatureStore,
    Entity,
    FeatureSet,
    MaterializationSettings,
    FeatureSetSpec,
)
from azure.ai.ml.constants import AssetTypes

subscription_id = "a485bb50-61aa-4b2f-bc7f-b6b53539b9d3"
resource_group = "rg-60104832"
workspace = "lab5mri"

# Authenticate
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id,
    resource_group,
    workspace,
)

# ---------------------------
# 1. CREATE ENTITY
# ---------------------------
entity = Entity(
    name="tumorimage",
    index_columns=[{"name": "image_id", "type": "string"}],
    version="1",
)

ml_client.feature_stores.begin_create_or_update(entity).result()
print("Entity created: tumorimage")


# ---------------------------
# 2. CREATE FEATURE SET
# ---------------------------
feature_set = FeatureSet(
    name="tumor_features",
    version="1",
    description="Extracted MRI image features",
    entities=["tumorimage"],

    spec=FeatureSetSpec(
        path="azureml://datastores/workspaceblobstore/paths/silver/features.parquet",
        features=[
            {"name": "f1", "type": "float"},
            {"name": "f2", "type": "float"},
            {"name": "f3", "type": "float"},
            # Add more features as needed...
        ],
    ),

    materialization_settings=MaterializationSettings(
        schedule={"cron": "0 * * * *"},  # every hour
        spark_configuration={"driver_cores": 2, "driver_memory": "4g"},
    ),
)

ml_client.feature_sets.begin_create_or_update(feature_set).result()
print("Feature set created: tumor_features")
