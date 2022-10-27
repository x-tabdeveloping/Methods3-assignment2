# %%
import arviz as az
import pymc as pm
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from utils.model import create_model
from utils.simulation import (
    simulate_studies,
    power_analysis,
    bias_analysis,
)

# %%
studies = simulate_studies(n_studies=100)

# %%
studies
# %%
model = create_model(studies)
# %% Prior predictive check
trace = pm.sample_prior_predictive(samples=4000, model=model)
az.plot_ppc(trace, group="prior")

# %% Sampling posterior
trace.extend(pm.sample(model=model))

# %% Trace plot
az.plot_trace(trace)

# %% Posterior predictive check
trace.extend(pm.sample_posterior_predictive(trace, model=model))
az.plot_ppc(trace, group="posterior")
# %% Prior-posterior update
az.plot_dist_comparison(trace, combine_dims={"study_id_levels"})

# %% Summary
az.summary(trace)
# %% Checking the effect of publication-bias
published_model = create_model(published_studies)
published_trace = pm.sample(model=published_model)
az.summary(published_trace)

# %% Extracting effects from the models
effect = trace.posterior.population_level_effect.to_numpy().flatten()
published_effect = (
    published_trace.posterior.population_level_effect.to_numpy().flatten()
)

# %% Plotting effect difference
hist = go.Histogram(
    x=effect,
    # opacity=0.4,
    marker_color="red",
    name="All data",
)
published_hist = go.Histogram(
    x=published_effect,
    # opacity=0.4,
    marker_color="blue",
    name="Published studies",
)
fig = go.Figure(data=(hist, published_hist))
fig.update_layout(
    title_text="Estimated mean effect size in all studies vs. only published ones"
)
fig.show()

# %% Power analysis
power = power_analysis(
    sample_sizes=[10, 15, 20, 30, 50, 75, 100, 200],
    create_model=create_model,
    publication_bias=False,
)
power.to_csv("results/power_analysis.csv")

# %%
px.scatter(
    power,
    x="trial",
    facet_col="n_studies",
    facet_col_wrap=2,
    y="mean",
    error_y=power["hdi_97%"] - power["mean"],
    error_y_minus=power["mean"] - power["hdi_3%"],
    height=800,
).add_hline(y=0.4, line_color="red").update_xaxes(type="category")
# %% Bias analysis
bias = bias_analysis(
    bias_values=np.arange(0.0, 1.0, 0.1),
    create_model=create_model,
    sample_size=100,
)
bias.to_csv("results/bias_analysis.csv")
# %%
px.scatter(
    bias,
    x="trial",
    y="mean",
    facet_col="bias",
    facet_col_wrap=2,
    error_y=bias["hdi_97%"] - bias["mean"],
    error_y_minus=bias["mean"] - bias["hdi_3%"],
    height=800,
).add_hline(y=0.4, line_color="red").update_xaxes(type="category")
# %%
