"""
Trends page — Historical aging population + hospital admissions time series.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.data_pipeline import load_population, load_hospital_admissions, build_features, load_eldercare, load_dementia_gtp, assign_infrastructure_to_subzones, load_clinics
from utils.clustering import run_clustering, TIER_ORDER, CLUSTER_COLORS

# Load trends_forecast_tab directly by file path — avoids ModuleNotFoundError
# when views/ has no __init__.py and can't be imported as a package.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "trends_forecast_tab",
    os.path.join(os.path.dirname(__file__), "trends_forecast_tab.py"),
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
render_forecast_tab = _mod.render_forecast_tab

@st.cache_data(show_spinner=False)
def get_trend_data():
    pop = load_population()
    hosp = load_hospital_admissions()
    return pop, hosp


# ── Hospital data helpers ──────────────────────────────────────────────────────

ACUTE_HOSPITALS = [
    "Alexandra Hospital",
    "Changi General Hospital",
    "Khoo Teck Puat Hospital",
    "National University Hospital",
    "Ng Teng Fong General Hospital",
    "Sengkang General Hospital",
    "Singapore General Hospital",
    "Tan Tock Seng Hospital",
    "Woodlands Health Campus",
]

SPECIALTY_HOSPITALS = [
    "Kandang Kerbau Women's & Children's Hospital",
    "National Heart Centre",
    "Institute Of Mental Health / Woodbridge Hospital",
    "National Centre For Infectious Diseases",
]


def prep_wide_hosp(hosp: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format hosp data to wide (date × hospital columns)."""
    total = hosp[hosp["DataSeries"] == "Public Sector Hospital Admissions"].copy()
    total = total[["date", "admissions"]].rename(columns={"admissions": "Public Sector Hospital Admissions"})

    hospitals = hosp[hosp["DataSeries"] != "Public Sector Hospital Admissions"].copy()
    wide = hospitals.pivot_table(index="date", columns="DataSeries", values="admissions", aggfunc="first")
    wide.columns = [c.strip() for c in wide.columns]
    wide = wide.reset_index().merge(total, on="date", how="left")
    wide["Year"]  = wide["date"].dt.year
    wide["Month"] = wide["date"].dt.month
    return wide


# ── Transition risk: 2025 → 2030 projection ────────────────────────────────

def get_projected_clustering(pop: pd.DataFrame, target_year: int = 2030) -> pd.DataFrame:
    """
    Forecast senior_pop_{target_year} per subzone via Prophet, recompute features,
    re-run clustering, and return a merged table comparing 2025 vs projected tiers.

    Other features (infrastructure density, distances) are held constant — this
    isolates the effect of demographic change alone on risk tier membership.

    Results are cached to disk (data/pipeline_cache/projection_{target_year}.csv) so the
    expensive per-subzone Prophet fitting only ever runs once, even across app
    restarts. No @st.cache_data here deliberately — hashing the large `pop`
    DataFrame on every call is itself costly; the disk-existence check below
    is the actual short-circuit and is checked first, before any hashing.
    Delete the cache file to force a recompute (e.g. after new SingStat data
    is added).
    """
    cache_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "pipeline_cache", f"projection_{target_year}.csv"
    )
    if "_proj_cache_status" not in st.session_state:
        st.session_state["_proj_cache_status"] = {}

    if os.path.exists(cache_path):
        st.session_state["_proj_cache_status"][target_year] = "disk"
        return pd.read_csv(cache_path)

    st.session_state["_proj_cache_status"][target_year] = "computed"

    from prophet import Prophet

    # ── Current (2025) clustered baseline ──────────────────────────────────
    feat_2025 = build_features(pop)
    ec = load_eldercare()
    dg = load_dementia_gtp()
    poly, hospitals = load_clinics()
    feat_2025 = assign_infrastructure_to_subzones(feat_2025, ec, dg, poly, hospitals)
    clustered_2025 = run_clustering(feat_2025)

    # ── Per-subzone yearly senior population series (2011–2025) ────────────
    senior_yearly = (
        pop[pop["AgeNum"] >= 65]
        .groupby(["PA", "SZ", "Time"])["Pop"].sum()
        .reset_index()
    )

    horizon = target_year - senior_yearly["Time"].max()
    projections = []

    for (pa, sz), grp in senior_yearly.groupby(["PA", "SZ"]):
        grp = grp.sort_values("Time")
        if len(grp) < 4 or grp["Pop"].sum() == 0:
            # Too sparse for Prophet — flat-line projection
            last_val = grp["Pop"].iloc[-1] if len(grp) else 0.0
            projections.append({"PA": pa, "SZ": sz, "senior_pop_proj": last_val})
            continue

        df_p = pd.DataFrame({
            "ds": pd.to_datetime(grp["Time"], format="%Y"),
            "y": grp["Pop"].astype(float),
        })
        try:
            m = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                        daily_seasonality=False, growth="linear")
            m.fit(df_p)
            future = m.make_future_dataframe(periods=horizon, freq="YS")
            fc = m.predict(future)
            proj_val = fc["yhat"].iloc[-1]
        except Exception:
            proj_val = grp["Pop"].iloc[-1]

        projections.append({"PA": pa, "SZ": sz, "senior_pop_proj": max(proj_val, 0.0)})

    proj_df = pd.DataFrame(projections)

    # ── Build projected feature set ─────────────────────────────────────────
    feat_proj = feat_2025.merge(proj_df, on=["PA", "SZ"], how="left")
    feat_proj["senior_pop_proj"] = feat_proj["senior_pop_proj"].fillna(feat_proj["senior_pop_2025"])

    # Recompute growth rate as CAGR from 2025 → target_year (annualised)
    years_fwd = target_year - 2025
    def cagr_fwd(end, start):
        if start <= 0 or years_fwd <= 0:
            return 0.0
        return (end / start) ** (1 / years_fwd) - 1

    feat_proj["senior_growth_rate_orig"] = feat_proj["senior_growth_rate"]
    feat_proj["senior_growth_rate"] = feat_proj.apply(
        lambda r: cagr_fwd(r["senior_pop_proj"], r["senior_pop_2025"]), axis=1
    ).clip(-0.3, 0.5)

    # Swap in projected senior population (infra/distance features held constant)
    feat_proj["senior_pop_2025_orig"] = feat_proj["senior_pop_2025"]
    feat_proj["senior_pop_2025"] = feat_proj["senior_pop_proj"]

    clustered_proj = run_clustering(feat_proj)

    # ── Merge 2025 vs projected tiers ────────────────────────────────────────
    # clustered_proj["senior_pop_2025"] now holds the *projected* value (swapped above)
    merged = clustered_2025[["PA", "SZ", "cluster_label", "senior_pop_2025"]].rename(
        columns={"cluster_label": "tier_2025", "senior_pop_2025": "senior_pop_2025_val"}
    ).merge(
        clustered_proj[["PA", "SZ", "cluster_label", "senior_pop_2025"]]
        .rename(columns={"cluster_label": f"tier_{target_year}", "senior_pop_2025": "senior_pop_proj_val"}),
        on=["PA", "SZ"], how="inner",
    )

    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        merged.to_csv(cache_path, index=False)
    except OSError as e:
        st.session_state["_proj_cache_status"][target_year] = f"save_failed: {e}"
    return merged


def render_transition_tab(pop: pd.DataFrame, target_year: int = 2030):
    """Identify subzones projected to move Emerging Pressure → High Pressure."""
    st.subheader(f"Projected Risk Tier Transitions by {target_year}")
    st.caption(
        "Per-subzone elderly population is forecast with Prophet using 2011–2025 "
        f"yearly data. Growth rate is recomputed from the {target_year} projection; "
        "infrastructure and distance features are held constant. Subzones are then "
        "re-clustered to identify tier shifts driven purely by demographic change."
    )

    with st.spinner(f"Forecasting subzone populations to {target_year}…"):
        merged = get_projected_clustering(pop, target_year)

    status = st.session_state.get("_proj_cache_status", {}).get(target_year)
    if status == "computed":
        st.caption("Computed this run and saved to disk — should load instantly on next restart.")
    elif status == "disk":
        st.caption(f"Loaded from cached results (data/pipeline_cache/projection_{target_year}.csv)")
    elif isinstance(status, str) and status.startswith("save_failed"):
        st.warning(
            f"Computed results this run, but could not write the cache file to disk ({status}). "
            "This means the forecast will re-run on every restart. Check write permissions "
            "on data/pipeline_cache/, or that the deployment filesystem isn't read-only/ephemeral."
        )

    tier_2025_col = "tier_2025"
    tier_proj_col = f"tier_{target_year}"

    transitions = merged[
        (merged[tier_2025_col] == "Emerging Pressure") &
        (merged[tier_proj_col] == "High Pressure")
    ].copy()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#E67E22,#C0392B)">
            <h3>{len(transitions)}</h3>
            <p>Subzones: Emerging → High Pressure</p></div>""", unsafe_allow_html=True)
    with c2:
        n_emerging = (merged[tier_2025_col] == "Emerging Pressure").sum()
        st.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#D4A017,#E67E22)">
            <h3>{n_emerging}</h3>
            <p>Currently Emerging Pressure</p></div>""", unsafe_allow_html=True)
    with c3:
        pct = (len(transitions) / n_emerging * 100) if n_emerging else 0
        st.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#1E3A5F,#2D5986)">
            <h3>{pct:.0f}%</h3>
            <p>Of Emerging Pressure Subzones at Risk</p></div>""", unsafe_allow_html=True)

    st.divider()

    if transitions.empty:
        st.info("No subzones are projected to transition from Emerging Pressure to High Pressure.")
        return

    display = transitions[["PA", "SZ", "senior_pop_2025_val", "senior_pop_proj_val"]].copy()
    display = display.sort_values("senior_pop_proj_val", ascending=False)
    display["senior_pop_2025_val"] = display["senior_pop_2025_val"].map("{:,.0f}".format)
    display["senior_pop_proj_val"] = display["senior_pop_proj_val"].map("{:,.0f}".format)
    display.columns = ["Planning Area", "Subzone", "Elderly (2025)", f"Elderly ({target_year}, projected)"]

    st.dataframe(display, use_container_width=True, hide_index=True)


def render():
    st.markdown("## Aging & Healthcare Trends")
    pop, hosp = get_trend_data()
    wide = prep_wide_hosp(hosp)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Senior Population", "Hospital Admissions", "Planning Area Breakdown",
        "Forecast",
    ])

    # ═══════════════════════════════════════════════════════════════════
    # TAB 1: SENIOR POPULATION
    # ═══════════════════════════════════════════════════════════════════
    with tab1:
        yearly_senior = (
            pop[pop["AgeNum"] >= 65].groupby("Time")["Pop"].sum().reset_index(name="Senior Population")
        )
        yearly_total = pop.groupby("Time")["Pop"].sum().reset_index(name="Total Population")
        yearly = yearly_senior.merge(yearly_total, on="Time")
        yearly["Aging Index (%)"] = yearly["Senior Population"] / yearly["Total Population"] * 100

        col1, col2 = st.columns(2)
        with col1:
            fig = px.area(yearly, x="Time", y="Senior Population",
                          title="Senior Population (65+) 2011–2025",
                          color_discrete_sequence=["#764ba2"],
                          labels={"Time": "Year", "Senior Population": "Residents"})
            fig.update_traces(line_width=2.5, fillcolor="rgba(118,75,162,0.15)")
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#F0F0F0"))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.line(yearly, x="Time", y="Aging Index (%)",
                           title="National Aging Index (%) 2011–2025",
                           color_discrete_sequence=["#E74C3C"], markers=True)
            fig2.update_traces(line_width=2.5, marker_size=7)
            fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               xaxis=dict(showgrid=False),
                               yaxis=dict(gridcolor="#F0F0F0", ticksuffix="%"))
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Age Structure — 2025 vs 2015")
        frames = []
        for yr in [2015, 2025]:
            tmp = pop[pop["Time"] == yr].groupby("AgeNum")["Pop"].sum().reset_index()
            tmp["Year"] = yr
            frames.append(tmp)
        combined = pd.concat(frames)
        fig3 = px.line(combined, x="AgeNum", y="Pop", color="Year",
                       title="Population Age Distribution: 2015 vs 2025",
                       color_discrete_map={2015: "#5B8DB8", 2025: "#E74C3C"},
                       labels={"AgeNum": "Age", "Pop": "Population"})
        fig3.add_vline(x=65, line_dash="dash", line_color="gray",
                       annotation_text="65 (Senior threshold)", annotation_position="top right")
        fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           xaxis=dict(showgrid=False, range=[0, 95]),
                           yaxis=dict(gridcolor="#F0F0F0"))
        st.plotly_chart(fig3, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════
    # TAB 2: HOSPITAL ADMISSIONS (full EDA suite)
    # ═══════════════════════════════════════════════════════════════════
    with tab2:

        # ── 2a: Total admissions + 12-month moving average ────────────
        st.subheader("Total Public Sector Admissions (1987–2026)")
        total_df = wide.dropna(subset=["Public Sector Hospital Admissions"]).copy()
        total_df["MA12"] = total_df["Public Sector Hospital Admissions"].rolling(12).mean()

        fig_tot = go.Figure()
        fig_tot.add_trace(go.Scatter(
            x=total_df["date"], y=total_df["Public Sector Hospital Admissions"],
            name="Monthly", line=dict(color="#AEC6CF", width=1.2), opacity=0.7,
        ))
        fig_tot.add_trace(go.Scatter(
            x=total_df["date"], y=total_df["MA12"],
            name="12-Month Moving Avg", line=dict(color="#E74C3C", width=2.2),
        ))
        # Annotate COVID dip
        fig_tot.add_vrect(x0="2020-02-01", x1="2021-06-01",
                          fillcolor="rgba(231,76,60,0.08)", line_width=0,
                          annotation_text="COVID-19", annotation_position="top left",
                          annotation_font_size=11)
        fig_tot.update_layout(
            plot_bgcolor="white", paper_bgcolor="white", height=360,
            xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#F0F0F0"),
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_tot, use_container_width=True)
        st.info(
            "**Long-run trend**: Admissions more than doubled from ~20,000/month in 1987 to ~46,000 by 2025, "
            "closely tracking Singapore's ageing demographics. The 65+ cohort went from 7% (1999) to 15%+ (2025) "
            "of the population — and between 2009–2014 alone, hospital admission growth was concentrated entirely in this age group."
        )

        # ── 2b: YoY % change ─────────────────────────────────────────
        st.subheader("Year-on-Year Admissions Growth")
        annual = (
            total_df[total_df["Year"] < 2026]
            .groupby("Year")["Public Sector Hospital Admissions"].sum()
            .reset_index(name="Total")
        )
        annual["YoY%"] = annual["Total"].pct_change() * 100
        fig_yoy = go.Figure()
        colors = ["#E74C3C" if v < 0 else "#27AE60" for v in annual["YoY%"].fillna(0)]
        fig_yoy.add_trace(go.Bar(x=annual["Year"], y=annual["YoY%"], marker_color=colors, name="YoY%"))
        fig_yoy.add_hline(y=0, line_color="gray", line_width=0.8)
        fig_yoy.update_layout(
            plot_bgcolor="white", paper_bgcolor="white", height=300,
            xaxis=dict(showgrid=False, dtick=5),
            yaxis=dict(gridcolor="#F0F0F0", ticksuffix="%"),
        )
        st.plotly_chart(fig_yoy, use_container_width=True)

        # ── 2c: Seasonal pattern ──────────────────────────────────────
        st.subheader("Seasonal Pattern: Average Admissions by Month")
        month_avg = (
            total_df.groupby("Month")["Public Sector Hospital Admissions"].mean().reset_index()
        )
        month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        month_avg["Month Label"] = month_avg["Month"].apply(lambda m: month_labels[m-1])

        fig_season = go.Figure()
        fig_season.add_trace(go.Bar(
            x=month_avg["Month Label"], y=month_avg["Public Sector Hospital Admissions"],
            marker_color=["#E74C3C" if m == 2 else "#5B8DB8" for m in month_avg["Month"]],
            name="Avg Admissions",
        ))
        fig_season.update_layout(
            plot_bgcolor="white", paper_bgcolor="white", height=320,
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="#F0F0F0"),
        )
        st.plotly_chart(fig_season, use_container_width=True)
        st.info(
            "**February dip**: Consistently the lowest month due to (1) fewer calendar days and "
            "(2) Chinese New Year — patients prefer to be discharged home before the holiday, "
            "causing a pre-CNY discharge surge in late January and delayed admissions into March."
        )

        # ── 2d: Per-hospital trends ───────────────────────────────────
        st.subheader("Admissions by Acute Hospital")
        available_acute = [h for h in ACUTE_HOSPITALS if h in wide.columns]
        selected_hosp = st.multiselect(
            "Select hospitals", available_acute, default=available_acute,
            key="hosp_select"
        )
        if selected_hosp:
            fig_hosp = go.Figure()
            for h in selected_hosp:
                smoothed = wide[h].rolling(6).mean()
                fig_hosp.add_trace(go.Scatter(
                    x=wide["date"], y=smoothed, name=h, mode="lines", line=dict(width=1.8),
                ))
            fig_hosp.update_layout(
                plot_bgcolor="white", paper_bgcolor="white", height=400,
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor="#F0F0F0"),
                legend=dict(orientation="h", y=-0.3, font_size=11),
            )
            st.plotly_chart(fig_hosp, use_container_width=True)

            # Last-12-months share bar
            st.subheader("Volume Share — Last 12 Months")
            latest = wide.tail(12)
            shares = {h: latest[h].sum() for h in selected_hosp if latest[h].notna().any()}
            shares_s = pd.Series(shares).sort_values()
            fig_share = px.bar(
                shares_s.reset_index(), x=0, y="index", orientation="h",
                color=0, color_continuous_scale="Blues",
                labels={"index": "Hospital", 0: "Admissions"},
                title="Total Admissions — Last 12 Months",
            )
            fig_share.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                coloraxis_showscale=False, height=320,
                xaxis=dict(gridcolor="#F0F0F0"), yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_share, use_container_width=True)
            st.info(
                "**Volume hierarchy**: SGH, TTSH, NUH and KKH dominate volume as the oldest "
                "tertiary referral centres handling the highest-complexity cases. "
                "Sengkang GH and Ng Teng Fong are newer facilities whose ramp-up is visible post-2015."
            )

        # ── 2e: Correlation matrix ────────────────────────────────────
        st.subheader("Inter-Hospital Correlation")
        if len(selected_hosp) >= 2:
            corr = wide[selected_hosp].corr()
            abbr = ["".join([w[0] for w in h.split()]) for h in selected_hosp]
            # Fix SGH / Sengkang clash
            abbr = [a if a != "SGH" or h == "Singapore General Hospital" else "SKGH"
                    for a, h in zip(abbr, selected_hosp)]
            fig_corr = go.Figure(go.Heatmap(
                z=corr.values, x=abbr, y=abbr,
                colorscale="RdYlGn", zmin=-1, zmax=1,
                text=corr.round(2).values, texttemplate="%{text}",
                hoverongaps=False,
            ))
            fig_corr.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig_corr, use_container_width=True)
            st.info(
                "**Systemic risk**: High correlations (>0.8) mean hospitals peak simultaneously "
                "during island-wide events (flu outbreaks, haze, COVID surges). "
                "A cluster of correlated hospitals cannot buffer each other — "
                "system capacity must be planned for the aggregate, not the individual."
            )

    # ═══════════════════════════════════════════════════════════════════
    # TAB 3: PLANNING AREA BREAKDOWN
    # ═══════════════════════════════════════════════════════════════════
    with tab3:

        st.subheader("Top Planning Areas by Senior Population (2025)")
        pa_senior = (
            pop[(pop["Time"] == 2025) & (pop["AgeNum"] >= 65)]
            .groupby("PA")["Pop"].sum().reset_index(name="Senior Pop 2025")
            .sort_values("Senior Pop 2025", ascending=False).head(20)
        )
        fig_top = px.bar(
            pa_senior, x="Senior Pop 2025", y="PA", orientation="h",
            color="Senior Pop 2025", color_continuous_scale="RdPu",
            labels={"PA": "Planning Area", "Senior Pop 2025": "Seniors (65+)"},
        )
        fig_top.update_layout(
            plot_bgcolor="white", paper_bgcolor="white", height=520,
            yaxis=dict(autorange="reversed"), coloraxis_showscale=False,
        )
        st.plotly_chart(fig_top, use_container_width=True)

        # ── PA Growth chart with anomaly handling + PA selector ───────
        st.subheader("Senior Population Growth by Planning Area (2015→2025)")
        st.caption(
            "Anomaly handling: growth rates are capped at ±100% and PAs with "
            "<200 seniors in 2015 are excluded to avoid distortion from small base effects."
        )

        pa_2015 = (
            pop[(pop["Time"] == 2015) & (pop["AgeNum"] >= 65)]
            .groupby("PA")["Pop"].sum().reset_index(name="pop_2015")
        )
        pa_2025 = (
            pop[(pop["Time"] == 2025) & (pop["AgeNum"] >= 65)]
            .groupby("PA")["Pop"].sum().reset_index(name="pop_2025")
        )
        pa_growth = pa_2015.merge(pa_2025, on="PA")

        # Anomaly handling
        # 1. Exclude tiny base populations (small number fluctuations → huge %)
        pa_growth = pa_growth[pa_growth["pop_2015"] >= 200]
        # 2. Compute CAGR (more meaningful than raw %)
        pa_growth["cagr"] = (
            (pa_growth["pop_2025"] / pa_growth["pop_2015"]) ** (1/10) - 1
        ) * 100
        # 3. Winsorise outliers beyond ±15% CAGR (rare; data artefacts from boundary changes)
        pa_growth["cagr_display"] = pa_growth["cagr"].clip(-15, 15)
        pa_growth = pa_growth.sort_values("cagr_display", ascending=False)

        all_pas = sorted(pa_growth["PA"].unique())
        selected_pas = st.multiselect(
            "Select Planning Areas to display",
            options=all_pas,
            default=all_pas[:20],
            key="pa_growth_select",
        )
        pa_plot = pa_growth[pa_growth["PA"].isin(selected_pas)] if selected_pas else pa_growth

        fig_growth = px.bar(
            pa_plot, x="PA", y="cagr_display",
            color="cagr_display", color_continuous_scale="RdYlGn_r",
            labels={"PA": "Planning Area", "cagr_display": "CAGR (%)"},
            title="10-Year Senior Population CAGR by Planning Area (2015→2025)",
        )
        fig_growth.add_hline(y=0, line_color="gray", line_width=0.8)
        fig_growth.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(tickangle=-35, showgrid=False),
            yaxis=dict(gridcolor="#F0F0F0", ticksuffix="%"),
            coloraxis_showscale=False, height=420,
        )
        st.plotly_chart(fig_growth, use_container_width=True)
        st.info(
            "**Reading this chart**: CAGR (Compound Annual Growth Rate) is used instead of raw % growth "
            "to make fast-growing and slow-growing PAs comparable on the same scale. "
            "A CAGR of +5% means the senior population doubles every ~14 years. "
            "PAs with fewer than 200 seniors in 2015 are excluded as their growth percentages "
            "are misleadingly large from a small base."
        )

    # ═══════════════════════════════════════════════════════════════════
    # TAB 4: FORECAST (incl. Transition Risk)
    # ═══════════════════════════════════════════════════════════════════
    with tab4:
        render_forecast_tab(pop)
        st.divider()
        render_transition_tab(pop, target_year=2030)
