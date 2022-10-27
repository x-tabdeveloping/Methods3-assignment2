# %% Importing packages
import arviz as az
import numpy as np
import pandas as pd
import plotly.express as px
import pymc as pm

from utils.data import clean_data
from utils.model import create_model
from utils.plots import funnel_plot

# %% Loading data
data = pd.read_excel("data.xlsx")
data.info()

# %% Cleaning data
data = clean_data(data)
data.to_csv("clean_data.csv")
data.info()

# %% Plotting sample sizes
n_studies = len(data.index)
sample_sizes = pd.DataFrame(
    {
        "Diagnosis": ["Control"] * n_studies + ["Schizophrenic"] * n_studies,
        "Sample size": pd.concat(
            (data.n_control, data.n_schizo), ignore_index=True
        ),
    }
)
px.box(sample_sizes, x="Diagnosis", y="Sample size").show()

#%%
(data.n_control + data.n_schizo).describe()

#%%
px.histogram(data.n_schizo / data.n_control)

# %% Plotting task distribution
px.histogram(data, x="task")

# %% Checking for publication bias with a funnel plot
funnel = funnel_plot(data, effect_col="cohens_d", error_col="se_cohens_d")
funnel.show()

# %% Fitting model on Cohen's d
model = create_model(data, effect_col="cohens_d", error_col="se_cohens_d")
trace = pm.sample_prior_predictive(model=model)
trace.extend(pm.sample(model=model))
trace.extend(pm.sample_posterior_predictive(trace, model=model))

# %%
summary = az.summary(trace)
summary

# %%
az.plot_forest(
    trace,
    var_names=["~population_level_effect"],
    combine_dims={"chain", "draw"},
    combined=True,
    r_hat=True,
    ess=True,
)
# %%
az.plot_ppc(trace, group="posterior")
# %%
