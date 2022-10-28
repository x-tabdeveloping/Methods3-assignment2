"""Module containing statistical utilities."""
import numpy as np


def cohens_d(mu_1, mu_2, sd_1, sd_2):
    """Computes Cohen's d effect size."""
    v1 = sd_1**2
    v2 = sd_2**2
    pooled_sd = np.sqrt((v1 + v2) / 2)
    d = (mu_2 - mu_1) / pooled_sd
    return d
    # return compute_measure("d", m1=mu_1, m2=mu_2, sd1=sd_1, sd2=sd_2, )


def se_cohens_d(d, n_1, n_2):
    """Computes standard error od Cohen's d."""
    # breaking up the formula
    n = n_1 + n_2
    a = n / (n_1 * n_2)
    b = d**2 / (2 * (n - 2))
    c = n / (n - 2)
    return (a + b) * c


def se_effect(sd_1, sd_2, n_1, n_2):
    """Computes standard error of differences of two means."""
    return np.sqrt(sd_1**2 / n_1 + sd_2**2 / n_2)


# def correlation_bias(effect, se):
#     v_star =
