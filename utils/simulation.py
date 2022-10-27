from math import sqrt
from typing import Callable, Iterable

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


def simulate_studies(n_studies: int = 100) -> pd.DataFrame:
    """Simulates the given number of studies.

    Parameters
    ----------
    n_studies: int, default 100
        Number of studies to simulate

    Returns
    -------
    DataFrame
        Data about each study.
    """
    records = []
    for study_id in range(n_studies):
        n_participants = int(max(np.random.normal(20, 10), 10))
        mu_effect_size = np.random.normal(0.4, 0.4)
        error = 0.8
        effects = np.random.normal(mu_effect_size, error, size=n_participants)
        mean_effect = np.mean(effects)
        sd_effect = np.std(effects)
        se = sd_effect / sqrt(n_participants)
        confidence = 1.96 * se
        low, high = mean_effect - confidence, mean_effect + confidence
        significance = not ((low < 0) and (high > 0))
        records.append(
            {
                "study_id": study_id,
                "n_participants": n_participants,
                "standard_error": se,
                "mean_effect": mean_effect,
                "significance": significance,
            }
        )
    return pd.DataFrame.from_records(records)


def simulate_publications(
    significance: pd.Series, bias: float = 0.0
) -> pd.Series:
    """Simulates publication bias based on publishing probabilities.

    Parameters
    ----------
    significance: Series of bool
        Sequence indicating which studies are significant.
    bias: float
        Bias towards significant studies.
        Non-significant studies will have probability of (1-bias)
        of being published.

    Returns
    -------
    Series of bool
        Sequence indicating which studies get published.
    """
    n_studies = len(significance.index)
    non_significant_published = np.random.binomial(1, 1 - bias)
    published = np.where(significance, 1, non_significant_published)
    published = pd.Series(published, index=significance.index)
    return published.astype(bool)


def power_analysis(
    sample_sizes: Iterable[int],
    create_model: Callable[[pd.DataFrame], pm.Model],
    n_trials: int = 10,
) -> pd.DataFrame:
    records = []
    for sample_size in sample_sizes:
        for trial in range(n_trials):
            data = simulate_studies(n_studies=sample_size)
            trace = pm.sample(model=create_model(data))
            summary = az.summary(trace)
            results = summary.loc["population_level_effect"].to_dict()  # type: ignore
            record = {"trial": trial, "n_studies": sample_size, **results}
            records.append(record)
    return pd.DataFrame.from_records(records)


def bias_analysis(
    bias_values: Iterable[float],
    create_model: Callable[[pd.DataFrame], pm.Model],
    sample_size: int = 200,
    n_trials: int = 10,
) -> pd.DataFrame:
    records = []
    for bias in bias_values:
        for trial in range(n_trials):
            data = simulate_studies(n_studies=sample_size)
            publications = simulate_publications(data.significance, bias=bias)
            data = data[publications]
            trace = pm.sample(model=create_model(data))
            summary = az.summary(trace)
            results = summary.loc["population_level_effect"].to_dict()  # type: ignore
            record = {"trial": trial, "bias": bias, **results}
            records.append(record)
    return pd.DataFrame.from_records(records)
