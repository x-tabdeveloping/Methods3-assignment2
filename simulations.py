"""Script for running simulations."""
# %%
import arviz as az
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pymc as pm
from matplotlib.pyplot import savefig

from utils.model import create_model
from utils.simulation import bias_analysis, power_analysis, simulate_studies

# %% Simulating studies
studies = simulate_studies(n_studies=100)

# %%
studies
# %% Creating model
model = create_model(studies)
# %% Prior predictive check
trace = pm.sample_prior_predictive(samples=4000, model=model)
az.plot_ppc(trace, group="prior")
savefig("plots/sim_prior_pred.png")

# %% Sampling posterior
trace.extend(pm.sample(model=model))

# %% Trace plot
az.plot_trace(trace)
savefig("plots/sim_trace.png")

# %% Posterior predictive check
trace.extend(pm.sample_posterior_predictive(trace, model=model))
az.plot_ppc(trace, group="posterior")
savefig("plots/sim_post_pred.png")
# %% Prior-posterior update
az.plot_dist_comparison(trace, combine_dims={"study_id_levels"})
savefig("plots/sim_update.png")

# %% Summary
print("Posterior sample summary")
print(az.summary(trace))
# %% Power analysis
power = power_analysis(
    sample_sizes=[10, 15, 20, 30, 50, 75, 100, 200],
    create_model=create_model,
    publication_bias=False,
)
power.to_csv("results/power_analysis.csv")

# %% Plotting power analysis results
power_plot = (
    px.scatter(
        power,
        x="trial",
        facet_col="n_studies",
        facet_col_wrap=2,
        y="mean",
        error_y=power["hdi_97%"] - power["mean"],
        error_y_minus=power["mean"] - power["hdi_3%"],
        height=800,
    )
    .add_hline(y=0.4, line_color="red")
    .update_xaxes(type="category")
)
power_plot.write_image("plots/power_analysis.png")
power_plot
# %% Bias analysis
bias = bias_analysis(
    bias_values=np.arange(0.0, 1.0, 0.1),
    create_model=create_model,
    sample_size=100,
)
bias.to_csv("results/bias_analysis.csv")
# %% Plotting bias analysis
bias_plot = (
    px.scatter(
        bias,
        x="trial",
        y="mean",
        facet_col="bias",
        facet_col_wrap=2,
        error_y=bias["hdi_97%"] - bias["mean"],
        error_y_minus=bias["mean"] - bias["hdi_3%"],
        height=800,
    )
    .add_hline(y=0.4, line_color="red")
    .update_xaxes(type="category")
)
bias_plot.write_image("plots/bias_analysis.png")
bias_plot
