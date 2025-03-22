import logging
from datetime import datetime
from typing import Any

import neptune
import optuna
import plotly.express as px
import polars as pl
import streamlit as st
from darts import TimeSeries
from darts.metrics import mape
from neptune.types import File

logger = logging.getLogger(__name__)


class OptimizationExperiment:
    def __init__(self, model_name: str, api_token: str = None, project: str = "your-workspace/your-project"):
        self.model_name = model_name
        self.run = neptune.init_run(
            project=project,
            api_token=api_token,
            name=f"{model_name}_optimization",
            tags=["hyperparameter-optimization", model_name],
        )
        # Initialize Polars DataFrame for trials
        self.trials_df = pl.DataFrame(
            schema={
                "trial_id": pl.Int32,
                "timestamp": pl.Datetime,
                "params": pl.Struct,
                "value": pl.Float64,
                "status": pl.Utf8,
            },
        )

    def log_trial(self, trial: optuna.Trial, score: float):
        """Log trial results using Polars."""
        # Create new trial record
        new_trial = pl.DataFrame(
            {
                "trial_id": [trial.number],
                "timestamp": [datetime.now()],
                "params": [trial.params],
                "value": [score],
                "status": [trial.state.name],
            },
        )

        # Append to trials DataFrame
        self.trials_df = pl.concat([self.trials_df, new_trial])

        # Log to Neptune
        self.run[f"trials/{trial.number}/params"] = trial.params
        self.run[f"trials/{trial.number}/value"] = score

    def log_validation_predictions(self, val_data: TimeSeries, predictions: TimeSeries):
        """Log validation predictions using Polars."""
        try:
            # Convert to Polars DataFrame
            df = pl.DataFrame(
                {
                    "timestamp": val_data.time_index,
                    "actual": val_data.values().flatten(),
                    "predicted": predictions.values().flatten(),
                },
            )

            # Calculate prediction metrics
            metrics = df.select(
                [
                    pl.col("actual").mean().alias("mean_actual"),
                    pl.col("predicted").mean().alias("mean_predicted"),
                    ((pl.col("actual") - pl.col("predicted")).abs()).mean().alias("mae"),
                    ((pl.col("actual") - pl.col("predicted")) ** 2).mean().sqrt().alias("rmse"),
                ],
            )

            # Log metrics to Neptune
            self.run["validation_metrics"] = metrics.to_dict(as_series=False)

            # Create visualization
            fig = px.line(
                df.to_pandas(),
                x="timestamp",
                y=["actual", "predicted"],
                title="Validation Predictions vs Actual Values",
            )
            self.run["visualizations/validation_predictions"].upload(File.as_html(fig))

        except Exception as e:
            logger.error(f"Error logging predictions: {str(e)}")

    def get_trial_summary(self) -> pl.DataFrame:
        """Get summary statistics of trials using Polars."""
        if len(self.trials_df) == 0:
            return pl.DataFrame()

        summary = self.trials_df.groupby("status").agg(
            [
                pl.count("trial_id").alias("count"),
                pl.col("value").mean().alias("mean_value"),
                pl.col("value").min().alias("best_value"),
                pl.col("value").std().alias("std_value"),
            ],
        )

        return summary

    def log_best_trial(self, study: optuna.Study):
        """Log best trial results and create visualizations using Polars."""
        # Get best trial data
        best_trial_data = pl.DataFrame(
            {
                "parameter": list(study.best_params.keys()),
                "value": list(study.best_params.values()),
            },
        )

        # Log to Neptune
        self.run["best_trial/params"] = study.best_params
        self.run["best_trial/value"] = study.best_value

        # Create trial history DataFrame
        trials_history = pl.DataFrame(
            {
                "trial_number": [t.number for t in study.trials],
                "value": [t.value if t.value is not None else float("nan") for t in study.trials],
            },
        )

        # Create and log optimization visualization
        try:
            # Parameter importance plot
            fig_importance = optuna.visualization.plot_param_importances(study)
            self.run["visualizations/param_importance"].upload(File.as_html(fig_importance))

            # Optimization history using Plotly + Polars
            fig_history = px.line(
                trials_history.to_pandas(),
                x="trial_number",
                y="value",
                title="Optimization History",
            )
            self.run["visualizations/optimization_history"].upload(File.as_html(fig_history))

        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")

    def stop(self):
        """Stop the experiment and log final summary."""
        # Log final trial summary
        final_summary = self.get_trial_summary()
        self.run["final_summary"] = final_summary.to_dict(as_series=False)
        self.run.stop()


def create_nbeats_objective(train_data: TimeSeries, val_data: TimeSeries, experiment: OptimizationExperiment):
    """Creates an objective function for N-BEATS model optimization."""

    def objective(trial):
        from backend.models.data.nbeats_model import NBEATSPredictor

        params = {
            "input_chunk_length": trial.suggest_int("input_chunk_length", 12, 48),
            "output_chunk_length": trial.suggest_int("output_chunk_length", 6, 24),
            "num_stacks": trial.suggest_int("num_stacks", 2, 8),
            "num_blocks": trial.suggest_int("num_blocks", 1, 4),
            "num_layers": trial.suggest_int("num_layers", 2, 4),
            "layer_widths": trial.suggest_int("layer_widths", 128, 512),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "n_epochs": trial.suggest_int("n_epochs", 20, 100),
        }

        try:
            model = NBEATSPredictor(**params)
            model.fit(train_data, val_series=val_data)

            predictions = model.predict(n=len(val_data))

            # Calculate metrics using Polars
            actual_values = pl.Series("actual", val_data.values().flatten())
            pred_values = pl.Series("predicted", predictions.values().flatten())

            metrics_df = pl.DataFrame(
                {
                    "actual": actual_values,
                    "predicted": pred_values,
                },
            )

            mape_score = mape(val_data, predictions)

            # Log trial
            experiment.log_trial(trial, mape_score)
            experiment.log_validation_predictions(val_data, predictions)

            return mape_score

        except Exception as e:
            logger.error(f"Trial failed: {str(e)}")
            return float("inf")

    return objective


def optimize_hyperparameters(
    train_data: TimeSeries,
    val_data: TimeSeries,
    model_name: str,
    n_trials: int = 100,
    timeout: int = None,
    neptune_api_token: str = None,
    neptune_project: str = None,
) -> dict[str, Any]:
    """Perform hyperparameter optimization using Optuna with Polars and Neptune tracking."""
    experiment = OptimizationExperiment(
        model_name=model_name,
        api_token=neptune_api_token,
        project=neptune_project,
    )

    study = optuna.create_study(
        direction="minimize",
        study_name=f"{model_name}_optimization",
        pruner=optuna.pruners.MedianPruner(),
    )

    if model_name == "N-BEATS":
        objective = create_nbeats_objective(train_data, val_data, experiment)
    else:
        raise ValueError(f"Model {model_name} not supported for optimization")

    with st.spinner(f"Optimizing {model_name} hyperparameters..."):
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

    # Log best results
    experiment.log_best_trial(study)

    # Convert trials to Polars DataFrame
    trials_df = experiment.trials_df

    results = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "best_trial": study.best_trial,
        "trials_dataframe": trials_df,
        "study": study,
        "neptune_run": experiment.run,
    }

    experiment.stop()

    return results
