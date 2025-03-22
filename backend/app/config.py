from pydantic import BaseSettings


class Settings(BaseSettings):
    # Application settings
    app_name: str = "Time Series Forecasting App"
    debug: bool = False
    api_prefix: str = "/api/v1"

    # Data settings
    default_dataset: str = "Air Passengers"
    available_datasets: list[str] = [
        "Air Passengers",
        "Monthly Milk Production (Incomplete)",
        "Electricity Consumption (Zurich)",
        "Upload CSV",
    ]

    # Model settings
    available_models: list[str] = ["N-BEATS", "Prophet", "TiDE", "Chronos", "TSMixer", "TFT"]
    default_model: str = "N-BEATS"
    model_sizes: list[str] = ["small", "medium", "large"]
    default_model_size: str = "small"

    # Forecasting settings
    default_forecast_horizon: int = 30
    max_forecast_horizon: int = 365

    # Training settings
    default_train_test_split: float = 0.8

    # Visualization settings
    max_points_to_display: int = 1000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
