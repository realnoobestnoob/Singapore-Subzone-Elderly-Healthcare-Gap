"""
views/trends_forecast_tab.py

Forecast tab for the Trends page.

INTEGRATION — in views/trends.py
─────────────────────────────────
1. Add import at top:
       from views.trends_forecast_tab import render_forecast_tab

2. Expand the tab list to include the new tab:
       tab1, tab2, tab3, tab4 = st.tabs([
           "👥 Senior Population",
           "🏥 Hospital Admissions",
           "📊 Planning Areas",
           "🔮 Forecast",
       ])

3. At the end of render(), add:
       with tab4:
           render_forecast_tab(pop)
   where `pop` is the DataFrame from load_population() already in scope.
"""

from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils.forecasting import (
    forecast_aging_index,
    forecast_senior_population,
    forecast_top_pa_seniors,
)

_PALETTE = [
    "#D62728","#E8B800","#1A8C3A",
    "#9467BD","#8C564B","#E377C2","#17BECF","#BCBD22","#F07000","#2BA8A8",
]


def render_forecast_tab(pop: pd.DataFrame) -> None:
    st.markdown("### 🔮 Senior Population Forecast")
    st.markdown(
        "Projections via **Prophet** (Meta / Bayesian structural time series), "
        "trained on SingStat data 2011–2025. Shaded band = 90% credible interval."
    )

    col_scope, col_horizon = st.columns([3, 2])
    with col_scope:
        scope = st.radio(
            "View",
            ["🇸🇬 National", "📍 By Planning Area", "📈 Aging Index"],
            horizontal=True, key="fc_scope",
        )
    with col_horizon:
        horizon = st.slider(
            "Forecast horizon (years beyond 2025)",
            min_value=5, max_value=10, value=10, step=1, key="fc_horizon",
        )

    st.divider()

    if scope == "🇸🇬 National":
        _render_national(pop, horizon)
    elif scope == "📍 By Planning Area":
        _render_by_pa(pop, horizon)
    else:
        _render_aging_index(pop, horizon)


def _render_national(pop, horizon):
    with st.spinner("Running Prophet model…"):
        hist, fc = forecast_senior_population(pop, level="national", horizon=horizon)
    if fc.empty:
        st.warning("Insufficient historical data for a national forecast.")
        return
    st.plotly_chart(
        _forecast_fig(hist, fc, "seniors", "yhat", "yhat_lower", "yhat_upper",
                      "Singapore — Senior Population (65+) Forecast",
                      "Senior Population", "#D62728"),
        use_container_width=True,
    )
    _kpi_row(int(hist["seniors"].iloc[-1]), fc, int(hist["year"].iloc[-1]))


def _render_by_pa(pop, horizon):
    top_pas = (
        pop[(pop["Time"] == 2025) & (pop["AgeNum"] >= 65)]
        .groupby("PA")["Pop"].sum()
        .nlargest(20).index.tolist()
    )
    col_sel, col_ci = st.columns([4, 1])
    with col_sel:
        selected = st.multiselect(
            "Planning Areas", options=top_pas, default=top_pas[:5],
            format_func=str.title, key="fc_pa_sel",
        )
    with col_ci:
        show_ci = st.checkbox("Show CI", value=True, key="fc_ci")

    if not selected:
        st.info("Select at least one Planning Area.")
        return

    with st.spinner(f"Forecasting {len(selected)} Planning Areas…"):
        all_results = forecast_top_pa_seniors(pop, top_n=20, horizon=horizon)
    results = {pa: v for pa, v in all_results.items() if pa in selected}
    if not results:
        st.warning("No forecast available for selected Planning Areas.")
        return

    fig = go.Figure()
    last_hist_year = None
    for i, (pa, (hist, fc)) in enumerate(results.items()):
        color = _PALETTE[i % len(_PALETTE)]
        label = pa.title()
        last_hist_year = int(hist["year"].max())
        fig.add_trace(go.Scatter(
            x=hist["year"], y=hist["seniors"], mode="lines", name=label,
            line=dict(color=color, width=2.5), legendgroup=label,
        ))
        fig.add_trace(go.Scatter(
            x=fc["year"], y=fc["yhat"], mode="lines",
            line=dict(color=color, width=2.5, dash="dash"),
            legendgroup=label, showlegend=False, name=f"{label} forecast",
        ))
        if show_ci:
            xb = pd.concat([fc["year"], fc["year"][::-1]])
            yb = pd.concat([fc["yhat_upper"], fc["yhat_lower"][::-1]])
            fig.add_trace(go.Scatter(
                x=xb, y=yb, fill="toself", fillcolor=color, opacity=0.13,
                line=dict(width=0), hoverinfo="skip",
                legendgroup=label, showlegend=False,
            ))

    if last_hist_year:
        fig.add_vline(
            x=last_hist_year + 0.5, line_dash="dot", line_color="#AAAAAA", line_width=1,
            annotation_text="◀ Historical  |  Forecast ▶",
            annotation_font_color="#888888", annotation_position="top",
        )
    fig.update_layout(
        title="Senior Population Forecast by Planning Area",
        xaxis_title="Year", yaxis_title="Senior Population (65+)",
        hovermode="x unified", height=460,
        legend=dict(orientation="v", x=1.02, y=1),
        margin=dict(r=170, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#EEEEEE", dtick=2),
        yaxis=dict(gridcolor="#EEEEEE"),
    )
    st.plotly_chart(fig, use_container_width=True)

    fc_end_yr = int(fc["year"].max())
    rows = []
    for pa, (hist, fc) in results.items():
        last = int(hist["seniors"].iloc[-1])
        proj = int(fc["yhat"].iloc[-1])
        rows.append({
            "Planning Area": pa.title(),
            "Seniors (2025)": f"{last:,}",
            f"Forecast ({fc_end_yr})": f"{proj:,}",
            "90% CI": f"{int(fc['yhat_lower'].iloc[-1]):,} – {int(fc['yhat_upper'].iloc[-1]):,}",
            "Growth": f"+{(proj-last)/last*100:.1f}%",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_aging_index(pop, horizon):
    col_lvl, col_pa = st.columns([2, 3])
    with col_lvl:
        ai_level = st.radio("Level", ["National", "Planning Area"],
                            horizontal=True, key="fc_ai_lvl")
    pa_filter = None
    if ai_level == "Planning Area":
        top_pas = (
            pop[(pop["Time"] == 2025) & (pop["AgeNum"] >= 65)]
            .groupby("PA")["Pop"].sum().nlargest(30).index.tolist()
        )
        with col_pa:
            pa_filter = st.selectbox("Planning Area", top_pas,
                                     format_func=str.title, key="fc_ai_pa")

    with st.spinner("Forecasting aging index…"):
        hist_ai, fc_ai = forecast_aging_index(pop, pa_filter=pa_filter, horizon=horizon)
    if fc_ai.empty:
        st.warning("Insufficient data for aging index forecast.")
        return

    label = pa_filter.title() if pa_filter else "Singapore"
    hist_pct = hist_ai.copy()
    hist_pct["aging_index"] = (hist_pct["aging_index"] * 100).round(2)
    fc_pct = fc_ai.copy()
    for col in ["ai_yhat","ai_lower","ai_upper"]:
        fc_pct[col] = (fc_pct[col] * 100).round(2)

    fig = _forecast_fig(
        hist_pct, fc_pct, "aging_index", "ai_yhat", "ai_lower", "ai_upper",
        f"Aging Index Forecast — {label}", "Aging Index (%)", "#F07000",
        y_ticksuffix="%",
    )
    fig.add_hline(
        y=20, line_dash="dot", line_color="#D62728", line_width=1.2,
        annotation_text="20% critical threshold",
        annotation_font_color="#D62728", annotation_position="top right",
    )
    st.plotly_chart(fig, use_container_width=True)

    last_val = hist_pct["aging_index"].iloc[-1]
    fc_val   = fc_pct["ai_yhat"].iloc[-1]
    above_20 = fc_pct[fc_pct["ai_yhat"] >= 20]
    k1, k2, k3 = st.columns(3)
    k1.metric(f"Aging Index ({int(hist_pct['year'].iloc[-1])})", f"{last_val:.1f}%")
    k2.metric(f"Forecast ({int(fc_pct['year'].iloc[-1])})", f"{fc_val:.1f}%",
              delta=f"+{fc_val - last_val:.1f} pp")
    if not above_20.empty:
        k3.metric("Reaches 20%", str(int(above_20["year"].iloc[0])),
                  delta="⚠️ Critical threshold", delta_color="inverse")
    else:
        k3.metric("Reaches 20%?", "Beyond forecast window")


def _forecast_fig(hist, fc, hist_col, yhat_col, lower_col, upper_col,
                  title, y_label, color, y_ticksuffix=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["year"], y=hist[hist_col], mode="lines+markers", name="Historical",
        line=dict(color=color, width=2.5), marker=dict(size=6, color=color),
    ))
    xb = pd.concat([fc["year"], fc["year"][::-1]])
    yb = pd.concat([fc[upper_col], fc[lower_col][::-1]])
    fig.add_trace(go.Scatter(
        x=xb, y=yb, fill="toself", fillcolor=color, opacity=0.15,
        line=dict(width=0), hoverinfo="skip", name="90% Credible Interval",
    ))
    fig.add_trace(go.Scatter(
        x=fc["year"], y=fc[yhat_col], mode="lines+markers", name="Forecast",
        line=dict(color=color, width=2.5, dash="dash"),
        marker=dict(size=6, color=color, symbol="circle-open"),
    ))
    # Bridge last historical → first forecast
    fig.add_trace(go.Scatter(
        x=[hist["year"].iloc[-1], fc["year"].iloc[0]],
        y=[hist[hist_col].iloc[-1], fc[yhat_col].iloc[0]],
        mode="lines", line=dict(color=color, width=1.5, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_vline(
        x=hist["year"].max() + 0.5,
        line_dash="dot", line_color="#BBBBBB", line_width=1,
        annotation_text="◀ Historical  |  Forecast ▶",
        annotation_font_color="#888888", annotation_position="top",
    )
    fig.update_layout(
        title=title, xaxis_title="Year", yaxis_title=y_label,
        yaxis_ticksuffix=y_ticksuffix,
        hovermode="x unified", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#EEEEEE", dtick=1),
        yaxis=dict(gridcolor="#EEEEEE"),
        margin=dict(t=60, b=40),
    )
    return fig


def _kpi_row(last_val, fc, last_year):
    fc_end_yr  = int(fc["year"].iloc[-1])
    fc_end_val = int(fc["yhat"].iloc[-1])
    fc_end_hi  = int(fc["yhat_upper"].iloc[-1])
    pct = (fc_end_val - last_val) / last_val * 100
    k1, k2, k3 = st.columns(3)
    k1.metric(f"Seniors ({last_year})", f"{last_val:,}")
    k2.metric(f"Forecast ({fc_end_yr})", f"{fc_end_val:,}", delta=f"+{pct:.1f}%")
    k3.metric("90% CI upper", f"{fc_end_hi:,}",
              delta=f"±{(fc_end_hi-fc_end_val)/fc_end_val*100:.1f}%", delta_color="off")
