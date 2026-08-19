from pydantic import (
    BaseModel,
    Field,
    field_validator
)


EXPECTED_FEATURES = 30


class PredictionRequest(BaseModel):

    features: list[float] = Field(
        ...,
        description=(
            "30 features from the "
            "Breast Cancer Wisconsin dataset"
        ),
        min_length=EXPECTED_FEATURES,
        max_length=EXPECTED_FEATURES
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, value):

        for feature in value:

            if not isinstance(
                feature,
                (int, float)
            ):

                raise ValueError(
                    "All features must be numeric"
                )

        return value


class PredictionResponse(BaseModel):

    prediction: int

    class_name: str

    probability: float

    model_version: str