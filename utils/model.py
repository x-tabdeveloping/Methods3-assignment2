import numpy as np
import pandas as pd
import pymc as pm


def create_model(
    data: pd.DataFrame,
    effect_col: str = "mean_effect",
    error_col: str = "standard_error",
) -> pm.Model:
    """Creates inference model based on the supplied data frame."""
    study_ids = np.arange(len(data.index))
    coords = {
        "study_id_levels": study_ids,
    }
    with pm.Model() as model:
        for name, value in coords.items():
            model.add_coord(name, value, mutable=True)  # type: ignore
        study_id = pm.MutableData("study_id", study_ids)
        study_se = pm.MutableData(error_col, data[error_col])
        mean_effect = pm.MutableData(
            effect_col,
            data[effect_col],
        )
        population_effect = pm.Normal(
            "population_level_effect", mu=0.0, sigma=0.5
        )
        individual_effect = pm.Normal(
            "individual_effect",
            mu=population_effect,
            sigma=0.2,
            dims=("study_id_levels"),
        )
        effect_size = pm.Normal(
            "effect_size",
            mu=individual_effect[study_id],
            sigma=study_se,
            observed=mean_effect,
            shape=mean_effect.shape,  # type: ignore
        )
    return model
