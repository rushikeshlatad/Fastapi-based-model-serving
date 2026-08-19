from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    app_name: str = "Breast Cancer Random Forest API"

    app_version: str = "1.0.0"

    model_path: str = (
        "models/breast_cancer_rf_pipeline.joblib"
    )

    metadata_path: str = (
        "models/model_metadata.joblib"
    )

    api_key: str = "development-secret"

    model_version: str = "1.0.0"

    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


settings = Settings()