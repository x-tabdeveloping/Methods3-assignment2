"""Module for data cleaning"""
import pandas as pd

from utils.stats import cohens_d, se_cohens_d, se_effect


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.rename(
        columns={
            "PITCH_F0SD_HC_M": "control_mean",
            "PITCH_F0SD_HC_SD": "control_sd",
            "PITCH_F0SD_SZ_M": "schizo_mean",
            "PITCH_F0SD_SZ_SD": "schizo_sd",
            "StudyID": "study_id",
            "SAMPLE_SIZE_SZ": "n_schizo",
            "SAMPLE_SIZE_HC": "n_control",
            "TYPE_OF_TASK": "task",
        }
    )
    data = data[
        [
            "control_mean",
            "control_sd",
            "schizo_mean",
            "schizo_sd",
            "study_id",
            "n_schizo",
            "n_control",
            "task",
        ]
    ]
    data = data.dropna(axis="index")
    data = data.assign(
        cohens_d=cohens_d(
            mu_1=data.control_mean,
            mu_2=data.schizo_mean,
            sd_1=data.control_sd,
            sd_2=data.schizo_sd,
        )
    )
    data = data.assign(
        se_cohens_d=se_cohens_d(
            d=data.cohens_d, n_1=data.n_control, n_2=data.n_schizo
        )
    )
    data = data.assign(
        effect=data.schizo_mean - data.control_mean,
        se_effect=se_effect(
            sd_1=data.control_sd,
            sd_2=data.schizo_sd,
            n_1=data.n_control,
            n_2=data.n_schizo,
        ),
    )
    return data
