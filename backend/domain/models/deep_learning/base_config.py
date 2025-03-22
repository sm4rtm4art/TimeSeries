from dataclasses import dataclass
from typing import Any

from backend.utils.model_utils import get_training_config


@dataclass
class DeepModelConfig:
    """Base configuration for deep learning models."""

    model_name: str
    input_chunk_length: int
    output_chunk_length: int
    batch_size: int
    n_epochs: int
    learning_rate: float
    accelerator: str
    precision: str
    force_dtype: Any

    @classmethod
    def from_model_name(cls, model_name: str) -> "DeepModelConfig":
        """Create config from model name using yaml configurations."""
        from backend.utils.config_loader import load_hardware_config, load_model_config

        model_config = load_model_config()
        hardware_config = load_hardware_config()

        common = model_config["models"]["common"]
        specific = model_config["models"].get(model_name.lower(), {})
        training = get_training_config()

        return cls(
            model_name=model_name,
            input_chunk_length=common["input_chunk_length"],
            output_chunk_length=common["output_chunk_length"],
            batch_size=common["batch_size"],
            n_epochs=common["n_epochs"],
            learning_rate=specific.get("learning_rate", common["learning_rate"]),
            accelerator=training["accelerator"],
            precision=training["precision"],
            force_dtype=training["force_dtype"],
        )
