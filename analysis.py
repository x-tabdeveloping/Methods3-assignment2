"""Script for conducting the analysis of the empirical data, can be run in VSCode interactive mode too"""
# %% Importing packages
import arviz as az
import numpy as np
import pandas as pd
import plotly.express as px
import pymc as pm
from matplotlib.pyplot import savefig

from utils.data import clean_data
from utils.model import create_model
from utils.plots import funnel_plot

# %% Loading data
print("Information on the raw data")
data = pd.read_excel("data.xlsx")
data.info()

# %% Cleaning data
print("Information on the raw data")
data = clean_data(data)
data.to_csv("clean_data.csv")
data.info()

#%% Stats on sample size
print("Sample size descriptive statistics:")
print((data.n_control + data.n_schizo).describe())

# %% Plotting task distribution
task_hist = px.histogram(data, x="task")
task_hist.write_image("plots/tasks.png")
task_hist.show()

# %% Checking for publication bias with a funnel plot
funnel = funnel_plot(data, effect_col="cohens_d", error_col="se_cohens_d")
funnel.write_image("plots/funnel.png")
funnel.show()

# %% Fitting model on Cohen's d
model = create_model(data, effect_col="cohens_d", error_col="se_cohens_d")
trace = pm.sample_prior_predictive(model=model)
trace.extend(pm.sample(model=model))
trace.extend(pm.sample_posterior_predictive(trace, model=model))

# %% Summarising posterior sample
summary = az.summary(trace)
print("Summary of the trace")
print(summary)

# %% Plotting individual effects
az.plot_forest(
    trace,
    var_names=["~population_level_effect"],
    combine_dims={"chain", "draw"},
    combined=True,
    r_hat=True,
    ess=True,
)
savefig("plots/individual_effects.png")
