import argparse
import os
from http import HTTPStatus
from typing import Dict

import ray
from fastapi import FastAPI
from ray import serve
from starlette.requests import Request

from madewithml import predict
from madewithml.config import MLFLOW_TRACKING_URI, logger, mlflow

# Define application
app = FastAPI(
    title="Made With ML",
    description="Classify machine learning projects.",
    version="0.1",
)


@serve.deployment(num_replicas="1", ray_actor_options={"num_cpus": 8, "num_gpus": 0})
@serve.ingress(app)
class ModelDeployment:
    def __init__(self, run_id: str, threshold: int = 0.9):
        """Initialize the model."""
        self.run_id = run_id
        self.threshold = threshold
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  # so workers have access to model registry
        best_checkpoint = predict.get_best_checkpoint(run_id=run_id)
        self.predictor = predict.TorchPredictor.from_checkpoint(best_checkpoint)

    @app.get("/")
    def _index(self) -> Dict:
        """Health check."""
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {},
        }
        return response

    @app.get("/run_id/")
    def _run_id(self) -> Dict:
        """Get the run ID."""
        return {"run_id": self.run_id}

    @app.post("/predict/")
    async def _predict(self, request: Request):
        data = await request.json()
        title = data.get("title", "")
        description = data.get("description", "")
        pred_extra = {"endpoint": "/predict/", "title_length": len(title), "description_length": len(description)}
        logger.info("Predict request received", extra={"extra_data": pred_extra})
        sample_ds = ray.data.from_items([{"title": title, "description": description, "tag": ""}])
        results = predict.predict_proba(ds=sample_ds, predictor=self.predictor)

        # Apply custom logic
        for i, result in enumerate(results):
            pred = result["prediction"]
            prob = result["probabilities"]
            if prob[pred] < self.threshold:
                results[i]["prediction"] = "other"

        logger.info(
            f"Prediction: {results[0]['prediction']} (confidence={max(results[0]['probabilities'].values()):.3f})",
            extra={"extra_data": {"prediction": results[0]}},
        )
        return {"results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", help="run ID to use for serving.")
    parser.add_argument("--threshold", type=float, default=0.9, help="threshold for `other` class.")
    args = parser.parse_args()
    ray.init(runtime_env={"env_vars": {"GITHUB_USERNAME": os.environ["GITHUB_USERNAME"]}})
    serve.run(ModelDeployment.bind(run_id=args.run_id, threshold=args.threshold))
