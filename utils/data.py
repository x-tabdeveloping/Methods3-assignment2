import pandas as pd
from pymare.effectsize import compute_measure

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
    smd, se_smd = compute_measure(
        "SMD",
        comparison=2,
        return_type="tuple",
        m1=data.control_mean,
        m2=data.schizo_mean,
        sd1=data.control_sd,
        sd2=data.schizo_sd,
        n1=data.n_control,
        n2=data.n_schizo,
    )
    data = data.assign(smd=smd, se_smd=se_smd)
    return data
