#!/usr/bin/env python
"""
Time Series Model Benchmark CLI

A comprehensive benchmarking tool for time series models with different hardware
accelerators. Compares performance across CPU, MPS, and CUDA (if available) for
various model architectures.

Usage:
    python -m backend.tools.benchmark --model nbeats --runs 3 \
        --accelerator cpu mps
    python -m backend.tools.benchmark --model all --data-size 300 --horizon 30
"""

import argparse
import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from tabulate import tabulate

# Enable importing from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Model mapping to support various models in the benchmark
MODEL_MAPPING = {
    "nbeats": "backend.domain.models.deep_learning.nbeats.NBEATSPredictor",
    "tide": "backend.domain.models.deep_learning.tide_model.TiDEPredictor",
    "nhits": "backend.domain.models.deep_learning.nhits_model.NHITSPredictor",
}

# Environment variables for specific model configurations
MODEL_ENV_VARS: dict[str, dict[str, dict[str, str]]] = {
    "nbeats": {
        "cpu": {},
        "mps": {"FORCE_MPS_FOR_NBEATS": "1"},
    },
    "tide": {
        "cpu": {},
        "mps": {},
    },
    "nhits": {
        "cpu": {},
        "mps": {"FORCE_MPS_FOR_NHITS": "1"},
    },
}


def generate_test_data(periods: int = 200, pattern: str = "seasonal") -> TimeSeries:
    """Generate synthetic time series data for testing.

    Args:
        periods: Number of time periods to generate
        pattern: Pattern type ('seasonal', 'random', 'trend')

    Returns:
        TimeSeries: Darts TimeSeries object with test data
    """
    dates = pd.date_range(start="2023-01-01", periods=periods, freq="D")

    if pattern == "seasonal":
        # Seasonal pattern with trend
        values = np.linspace(0, 10, periods) + 5 * np.sin(np.linspace(0, 6 * np.pi, periods))
        values = values.reshape(-1, 1)
    elif pattern == "trend":
        # Linear trend with noise
        values = np.linspace(0, 20, periods) + np.random.normal(0, 1, periods)
        values = values.reshape(-1, 1)
    else:
        # Random walk
        values = np.cumsum(np.random.normal(0, 1, (periods, 1)), axis=0)

    return TimeSeries.from_times_and_values(dates, values)


def get_available_accelerators() -> dict[str, bool]:
    """Detect available accelerators in the system."""
    return {
        "cpu": True,  # CPU is always available
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
    }


def get_model_class(model_name: str) -> Any:
    """Import and return the model class from the specified module."""
    if model_name not in MODEL_MAPPING:
        raise ValueError(f"Model '{model_name}' not supported. Available models: {list(MODEL_MAPPING.keys())}")

    module_path, class_name = MODEL_MAPPING[model_name].rsplit(".", 1)

    try:
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        return model_class
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import {model_name} model: {e}") from e


def set_env_for_model(model_name: str, accelerator: str) -> None:
    """Set environment variables needed for model+accelerator combinations."""
    if model_name in MODEL_ENV_VARS and accelerator in MODEL_ENV_VARS[model_name]:
        for key, value in MODEL_ENV_VARS[model_name][accelerator].items():
            os.environ[key] = value
            print(f"Set environment variable: {key}={value}")


def benchmark_model(
    model_name: str,
    accelerator: str,
    data_size: int = 200,
    horizon: int = 20,
    pattern: str = "seasonal",
) -> dict[str, str | float]:
    """Benchmark a single model with the specified accelerator.

    Args:
        model_name: Name of the model to benchmark
        accelerator: Accelerator to use ('cpu', 'mps', 'cuda')
        data_size: Size of the test data
        horizon: Forecast horizon
        pattern: Data pattern type

    Returns:
        Dict containing timing results
    """
    # Set environment variables for this model/accelerator combo
    set_env_for_model(model_name, accelerator)

    # Generate test data
    data = generate_test_data(periods=data_size, pattern=pattern)

    # Get model class and create instance
    model_class = get_model_class(model_name)

    # Initialize model
    start_time = time.time()
    model = model_class()
    init_time = time.time() - start_time

    # Print hardware configuration
    hw_config = getattr(model, "_get_hardware_config", lambda: {"accelerator": accelerator})()
    print(f"Model: {model_name}, Hardware: {hw_config}")

    # Train model
    start_time = time.time()
    model.train(data)
    train_time = time.time() - start_time

    # Generate forecast
    start_time = time.time()
    try:
        # Try using forecast method first
        if hasattr(model, "forecast"):
            model.forecast(horizon=horizon)
        # Fall back to predict method if forecast isn't available
        elif hasattr(model, "predict"):
            model.predict(n=horizon)
        else:
            print(f"Warning: No forecast/predict method found for {model_name}")
    except Exception as e:
        print(f"Error during forecasting: {e}")
    forecast_time = time.time() - start_time

    # Calculate total time
    total_time = init_time + train_time + forecast_time

    return {
        "model": model_name,
        "accelerator": accelerator,
        "actual_accelerator": hw_config.get("accelerator", accelerator),
        "init_time": init_time,
        "train_time": train_time,
        "forecast_time": forecast_time,
        "total_time": total_time,
    }


def run_benchmarks(
    models: list[str],
    accelerators: list[str],
    data_size: int = 200,
    horizon: int = 20,
    pattern: str = "seasonal",
    runs: int = 1,
) -> pd.DataFrame:
    """Run benchmarks for all specified models and accelerators.

    Args:
        models: List of model names to benchmark
        accelerators: List of accelerators to test
        data_size: Size of the test data
        horizon: Forecast horizon
        pattern: Data pattern type
        runs: Number of runs for each benchmark (for averaging)

    Returns:
        DataFrame with benchmark results
    """
    results = []
    available_accel = get_available_accelerators()

    # Check if specified accelerators are available
    for accel in accelerators[:]:  # Copy to avoid modifying during iteration
        if not available_accel.get(accel, False):
            print(f"Warning: Accelerator '{accel}' is not available. Skipping.")
            accelerators.remove(accel)

    # Resolve 'all' to all available models
    if "all" in models:
        models = list(MODEL_MAPPING.keys())

    # Run benchmarks
    for model_name in models:
        for accelerator in accelerators:
            print(f"\nBenchmarking {model_name} with {accelerator} ({runs} runs)")

            # Skip incompatible combinations
            if model_name == "tft" and accelerator == "mps":
                print("  Skipping TFT with MPS (requires 64-bit precision, MPS only supports 32-bit)")
                continue

            # Run multiple times and average
            model_results = []
            for run in range(1, runs + 1):
                print(f"  Run {run}/{runs}...")
                try:
                    run_result = benchmark_model(
                        model_name=model_name,
                        accelerator=accelerator,
                        data_size=data_size,
                        horizon=horizon,
                        pattern=pattern,
                    )
                    model_results.append(run_result)
                except Exception as e:
                    print(f"  Error benchmarking {model_name} with {accelerator}: {e}")
                    break

            # Calculate averages if we have results
            if model_results:
                # Average the timing results
                avg_result = {
                    "model": model_name,
                    "accelerator": accelerator,
                    "actual_accelerator": model_results[0]["actual_accelerator"],
                    "init_time": np.mean([r["init_time"] for r in model_results]),
                    "train_time": np.mean([r["train_time"] for r in model_results]),
                    "forecast_time": np.mean([r["forecast_time"] for r in model_results]),
                    "total_time": np.mean([r["total_time"] for r in model_results]),
                    "runs": len(model_results),
                }
                results.append(avg_result)

    return pd.DataFrame(results)


def format_results(results: list[dict[str, str | float]]) -> list[str]:
    """Format benchmark results for display.

    Args:
        results: List of benchmark result dictionaries

    Returns:
        List of strings containing formatted benchmark results
    """
    # First group by model
    model_groups = pd.DataFrame(results).groupby("model")

    output = []
    for model_name, model_df in model_groups:
        output.append(f"\n=== Model: {model_name} ===")

        # Create a table for this model
        model_table = []
        headers = ["Accelerator", "Init (s)", "Train (s)", "Forecast (s)", "Total (s)"]

        for _, row in model_df.iterrows():
            model_table.append(
                [
                    f"{row['accelerator']} ({row['actual_accelerator']})",
                    f"{row['init_time']:.2f}",
                    f"{row['train_time']:.2f}",
                    f"{row['forecast_time']:.2f}",
                    f"{row['total_time']:.2f}",
                ]
            )

        # Add to output
        output.append(tabulate(model_table, headers=headers, tablefmt="grid"))

        # Add speedup metrics if we have multiple accelerators
        if len(model_df) > 1:
            output.append("\nSpeedup relative to CPU:")
            cpu_row = model_df[model_df["accelerator"] == "cpu"]

            if not cpu_row.empty:
                cpu_time = cpu_row.iloc[0]["total_time"]

                for _, row in model_df[model_df["accelerator"] != "cpu"].iterrows():
                    accel_time = row["total_time"]
                    speedup = cpu_time / accel_time
                    speedup_str = "faster" if speedup > 1 else "slower"
                    output.append(f"  {row['accelerator']}: {speedup:.2f}x {speedup_str}")

    return output


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark time series models with different accelerators")

    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=["nbeats"],
        choices=list(MODEL_MAPPING.keys()) + ["all"],
        help="Model(s) to benchmark (use 'all' for all models)",
    )

    parser.add_argument(
        "--accelerator",
        type=str,
        nargs="+",
        default=["cpu"],
        choices=["cpu", "mps", "cuda"],
        help="Accelerator(s) to test",
    )

    parser.add_argument(
        "--data-size",
        type=int,
        default=200,
        help="Size of the test data (number of time periods)",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="Forecast horizon (number of periods to predict)",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="seasonal",
        choices=["seasonal", "trend", "random"],
        help="Pattern type for the test data",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs for each benchmark (for averaging)",
    )

    parser.add_argument(
        "--save",
        type=str,
        help="Save results to CSV file",
    )

    return parser.parse_args()


def print_results(args: argparse.Namespace) -> None:
    """Print benchmark parameters."""
    print(f"Benchmarking models: {args.model}")
    print(f"Accelerators: {args.accelerator}")
    print(f"Data size: {args.data_size}, Horizon: {args.horizon}, Pattern: {args.pattern}")
    print(f"Runs per benchmark: {args.runs}")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    print_results(args)

    # Run benchmarks
    results = run_benchmarks(
        models=args.model,
        accelerators=args.accelerator,
        data_size=args.data_size,
        horizon=args.horizon,
        pattern=args.pattern,
        runs=args.runs,
    )

    # Print results
    if len(results) > 0:
        print("\n=== BENCHMARK RESULTS ===")
        # Convert DataFrame to list of dictionaries with correct typing
        results_list = [dict(record.items()) for record in results.to_dict("records")]
        # Cast to the correct type for type checking
        typed_results = cast(list[dict[str, str | float]], results_list)
        formatted_results = format_results(typed_results)
        print("\n".join(formatted_results))

        # Save results if requested
        if args.save:
            results.to_csv(args.save, index=False)
            print(f"\nResults saved to {args.save}")
    else:
        print("\nNo benchmark results collected.")


if __name__ == "__main__":
    main()
