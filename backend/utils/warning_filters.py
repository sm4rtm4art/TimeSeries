"""Warning Filters Utility

This module provides functions to suppress common warnings from dependencies.
"""

import logging
import warnings

logger = logging.getLogger(__name__)


def suppress_sklearn_warnings() -> None:
    """Suppress common warnings from scikit-learn."""
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message="'force_all_finite' was renamed to 'ensure_all_finite'",
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module="sklearn",
    )
    logger.debug("Scikit-learn warnings suppressed")


def suppress_statsmodels_warnings() -> None:
    """Suppress common warnings from statsmodels."""
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="statsmodels",
    )
    warnings.filterwarnings(
        "ignore",
        message="Non-stationary starting autoregressive parameters",
    )
    logger.debug("Statsmodels warnings suppressed")


def suppress_all_warnings() -> None:
    """Suppress all common warnings from the dependencies we use."""
    suppress_sklearn_warnings()
    suppress_statsmodels_warnings()

    # General FutureWarnings from any source
    warnings.filterwarnings("ignore", category=FutureWarning)

    logger.info("All common warnings have been suppressed")
