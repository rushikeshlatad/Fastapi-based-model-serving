from pathlib import Path
from typing import Any

import joblib
import numpy as np


class ModelManager:

    def __init__(
        self,
        model_path: str,
        metadata_path: str
    ):

        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

        self.model: Any = None
        self.metadata: dict = {}

    def load_model(self) -> None:

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Model file not found: "
                f"{self.model_path}"
            )

        if not self.metadata_path.exists():

            raise FileNotFoundError(
                f"Metadata file not found: "
                f"{self.metadata_path}"
            )

        self.model = joblib.load(
            self.model_path
        )

        self.metadata = joblib.load(
            self.metadata_path
        )

    def predict(
        self,
        features: list[float]
    ):

        if self.model is None:

            raise RuntimeError(
                "Model is not loaded"
            )

        X = np.array(
            [features],
            dtype=float
        )

        prediction = self.model.predict(X)

        probability = None

        if hasattr(
            self.model,
            "predict_proba"
        ):

            probabilities = (
                self.model.predict_proba(X)
            )

            probability = probabilities[0]

        return prediction, probability

    def is_ready(self) -> bool:

        return self.model is not None

    def get_metadata(self) -> dict:

        return self.metadata