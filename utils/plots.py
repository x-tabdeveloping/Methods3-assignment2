"""Module containing plotting utilities"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def funnel_plot(
    data: pd.DataFrame, effect_col: str, error_col: str
) -> go.Figure:
    """Creates a funnel plot for assessing publication bias in meta-analyses.

    Parameters
    ----------
    data: DataFrame
        Data about the meta-analysis.
    effect_col: str
        Name of the column that contains the effects.
    error_col: str
        Name of the column that contains the standard errors of the effects.

    Returns
    -------
    Figure
        Funnel plot.
    """
    fig = px.scatter(data, x=effect_col, y=error_col)
    fig.update_yaxes(autorange="reversed")
    mean_effect = data[effect_col].mean()
    max_error = data[error_col].max()
    fig.add_shape(
        type="line", x0=mean_effect, x1=mean_effect, y0=max_error, y1=0
    )
    fig.add_shape(
        type="line",
        x0=mean_effect - max_error * 1.96,
        y0=max_error,
        x1=mean_effect,
        y1=0,
        line=dict(dash="dash"),
    )
    fig.add_shape(
        type="line",
        x0=mean_effect + max_error * 1.96,
        y0=max_error,
        x1=mean_effect,
        y1=0,
        line=dict(dash="dash"),
    )
    fig.add_shape(
        type="line",
        x0=mean_effect - max_error * 2.576,
        y0=max_error,
        x1=mean_effect,
        y1=0,
        line=dict(dash="dot"),
    )
    fig.add_shape(
        type="line",
        x0=mean_effect + max_error * 2.576,
        y0=max_error,
        x1=mean_effect,
        y1=0,
        line=dict(dash="dot"),
    )
    return fig
