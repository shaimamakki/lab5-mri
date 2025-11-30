import json
import os
import glob

import joblib
import numpy as np

model = None


def init():
    """Called once when the endpoint starts."""
    global model

    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    # Look for model.pkl inside the model directory
    candidates = glob.glob(os.path.join(model_dir, "**", "model.pkl"), recursive=True)
    if candidates:
        model_path = candidates[0]
    else:
        model_path = os.path.join(model_dir, "model.pkl")

    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)


def run(raw_data):
    """
    raw_data is expected to be a JSON string like:
    {
      "features": [0.1, 0.2, 0.3, ..., 20 values total]
    }
    """
    try:
        data = json.loads(raw_data)
        features = np.array(data["features"]).reshape(1, -1)

        pred = model.predict(features)[0]
        prob = (
            float(model.predict_proba(features)[0, 1])
            if hasattr(model, "predict_proba")
            else None
        )

        return {"prediction": int(pred), "probability": prob}
    except Exception as e:
        return {"error": str(e)}
