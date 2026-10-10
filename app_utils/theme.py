"""Colour palette and Plotly styling shared by every page."""
from __future__ import annotations

import plotly.graph_objects as go

TEAL = "#0D9488"      # primary / call / "today"
ORANGE = "#EA580C"    # secondary / put / "expiry"
INDIGO = "#4F46E5"    # tertiary / reference (analytic BSM)
SLATE = "#64748B"     # neutral / guides
GREEN = "#16A34A"
RED = "#DC2626"
BAND = "rgba(13,148,136,0.15)"
GRID = "rgba(100,116,139,0.18)"

KIND_COLOR = {"call": TEAL, "put": ORANGE}


def style_fig(fig: go.Figure, title: str | None = None, x: str | None = None, y: str | None = None,
              height: int = 420, legend: bool = True) -> go.Figure:
    """Apply the shared look: transparent background, light grid, legend above the plot."""
    layout: dict = dict(
        height=height,
        margin=dict(l=10, r=10, t=70 if (legend or title) else 20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        showlegend=legend,
        font=dict(size=13),
    )
    if title:  # never pass title=None: Plotly renders the literal text "undefined"
        layout["title"] = dict(text=title, x=0.0, y=0.97, yanchor="top", font=dict(size=16))
    if legend:
        layout["legend"] = dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0)
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, showline=True, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=True, zerolinecolor=SLATE, zerolinewidth=1, showline=True, linecolor=GRID)
    if x:
        fig.update_xaxes(title_text=x)
    if y:
        fig.update_yaxes(title_text=y)
    return fig


def vref(fig: go.Figure, x: float, name: str, color: str = SLATE, dash: str = "dash", **kw) -> None:
    """Vertical reference line with a legend entry (avoids annotation/legend collisions)."""
    fig.add_vline(x=x, line=dict(color=color, dash=dash, width=1.5), **kw)
    fig.add_scatter(x=[None], y=[None], mode="lines", line=dict(color=color, dash=dash, width=1.5), name=name, hoverinfo="skip")


def href(fig: go.Figure, y: float, name: str, color: str = SLATE, dash: str = "dash", **kw) -> None:
    """Horizontal reference line with a legend entry."""
    fig.add_hline(y=y, line=dict(color=color, dash=dash, width=1.5), **kw)
    fig.add_scatter(x=[None], y=[None], mode="lines", line=dict(color=color, dash=dash, width=1.5), name=name, hoverinfo="skip")


def log_axis(fig: go.Figure, axis: str, vals: list[float], labels: list[str] | None = None, **kw) -> None:
    """Log axis with clean, sparse tick labels instead of Plotly's 2-9 minor clutter."""
    upd = fig.update_xaxes if axis == "x" else fig.update_yaxes
    upd(type="log", tickmode="array", tickvals=vals, ticktext=labels or [f"{v:g}" for v in vals], **kw)


def decades(values, pad: int = 0) -> tuple[list[float], list[str]]:
    """Powers of ten spanning ``values`` - tick positions/labels for a log axis."""
    import math
    lo, hi = min(values), max(values)
    ks = range(math.floor(math.log10(lo)) - pad, math.ceil(math.log10(hi)) + 1 + pad)
    return [10.0**k for k in ks], [f"{10.0**k:g}" for k in ks]
