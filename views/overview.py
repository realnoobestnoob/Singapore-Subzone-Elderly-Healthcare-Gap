"""
Overview page — Key metrics and executive summary.
"""
import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.clustering import TIER_ORDER, CLUSTER_COLORS
from utils.shared_data import get_pipeline_data


def render():
    st.markdown('<p class="main-title">SgHealth-Optimize</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Regional Eldercare Demand & Healthcare Accessibility Gaps in Singapore</p>', unsafe_allow_html=True)

    data = get_pipeline_data()
    pop, clustered, ec, dg, poly, hospitals = (
        data["pop"], data["clustered"], data["ec"], data["dg"], data["poly"], data["hospitals"]
    )

    # ── KPI Row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    total_seniors = clustered["senior_pop_2025"].sum()
    total_pop = clustered["total_pop_2025"].sum()
    national_aging_idx = total_seniors / total_pop if total_pop > 0 else 0
    critical_zones = (clustered["cluster_label"] == "High Pressure").sum()
    total_infra = len(ec) + len(dg) + len(poly) + len(hospitals)

    with c1:
        st.markdown(f"""<div class="metric-card">
            <h3>{total_seniors:,.0f}</h3>
            <p>Seniors (65+) in 2025</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#f093fb,#f5576c)">
            <h3>{national_aging_idx:.1%}</h3>
            <p>National Aging Index</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#e74c3c,#c0392b)">
            <h3>{critical_zones}</h3>
            <p>High Pressure Subzones</p></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#43e97b,#38f9d7)">
            <h3>{total_infra}</h3>
            <p>Total Healthcare Nodes</p></div>""", unsafe_allow_html=True)

    st.divider()

    # ── Risk distribution summary ───────────────────────────────────────────
    st.subheader("Subzone Risk Distribution")
    dist = (
        clustered.groupby("cluster_label")
        .agg(
            Subzones=("SZ", "count"),
            Avg_Senior_Pop=("senior_pop_2025", "mean"),
            Avg_Growth_Rate=("senior_growth_rate", "mean"),
            Total_Seniors=("senior_pop_2025", "sum"),
        )
        .reset_index()
    )
    dist["_order"] = dist["cluster_label"].apply(
        lambda x: TIER_ORDER.index(x) if x in TIER_ORDER else 99
    )
    dist = dist.sort_values("_order").drop(columns="_order")
    dist["Avg_Senior_Pop"] = dist["Avg_Senior_Pop"].map("{:,.0f}".format)
    dist["Avg_Growth_Rate"] = dist["Avg_Growth_Rate"].map("{:.1%}".format)
    dist["Total_Seniors"]   = dist["Total_Seniors"].map("{:,.0f}".format)
    dist.columns = ["Risk Tier", "Subzones", "Avg No. of Elderly", "Avg Growth Rate", "Total Seniors"]
    st.dataframe(dist, use_container_width=True, hide_index=True)

    st.divider()

    # ── Per-cluster subzone tables ──────────────────────────────────────────
    st.subheader("Subzones by Risk Tier")
    active_tiers = [t for t in TIER_ORDER if t in clustered["cluster_label"].values]

    for tier in active_tiers:
        tier_df = (
            clustered[clustered["cluster_label"] == tier]
            .sort_values("senior_pop_2025", ascending=False)
            [["PA", "SZ", "senior_pop_2025", "senior_growth_rate", "total_pop_2025", "infra_density"]]
            .copy()
        )
        color = CLUSTER_COLORS.get(tier, "#95A5A6")
        with st.expander(f"{tier}  ({len(tier_df)} subzones)", expanded=(tier == "High Pressure")):
            st.markdown(
                f'<div style="height:4px;background:{color};border-radius:2px;'
                f'margin-bottom:10px;"></div>', unsafe_allow_html=True,
            )
            tier_df["senior_pop_2025"]   = tier_df["senior_pop_2025"].map("{:,.0f}".format)
            tier_df["senior_growth_rate"] = tier_df["senior_growth_rate"].map("{:+.1%}".format)
            tier_df["total_pop_2025"]     = tier_df["total_pop_2025"].map("{:,.0f}".format)
            tier_df["infra_density"]      = tier_df["infra_density"].map("{:.2f}".format)
            tier_df.columns = ["Planning Area", "Subzone", "Elderly (2025)", "Growth Rate", "Total Pop", "Infra Density"]
            st.dataframe(tier_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Methodology")
    with st.expander("How are risk clusters calculated?", expanded=False):
        st.markdown("""
| Feature | Weight | Description |
|---|---|---|
| **Dist. to Nearest Polyclinic** | 1.8 (highest) | Euclidean distance (km) to closest polyclinic — strongest predictor |
| **Infrastructure Density** | 1.2 | Weighted facility count per 1,000 residents — hospitals (5×), polyclinics (3×), eldercare & dementia GTPs (1×) |
| **Dist. to Nearest Hospital** | 1.1 | Euclidean distance (km) to closest public hospital |
| **Demand/Supply Ratio** | derived | (No. of Elderly ÷ 1,000) ÷ effective supply, discounted by distance penalty |
| **No. of Elderly** | 0.5 | Absolute 65+ resident count per subzone (2025) |
| **Senior Growth Rate** | 0.3 (lowest) | 10-year CAGR of 65+ population (2015→2025) |

**Agglomerative Clustering (Ward linkage)** with RobustScaler, k=3 fixed for interpretability (High Pressure / Emerging Pressure / Well-Served), silhouette ~0.72 vs 0.31 for KMeans. RobustScaler handles the lognormal skew in elderly counts and infra density; Ward linkage minimises within-cluster variance without random initialisation sensitivity. Weights derived from Random Forest + Permutation importance:

`risk_score = 1.8 × dist_poly + 1.2 × infra_density_penalty + 1.1 × dist_hospital + 0.5 × (elderly ÷ 1000) + 0.3 × growth_rate`

Distance to nearest polyclinic is the strongest predictor (RF importance 0.29, permutation 0.20), followed by infrastructure density and hospital access. Raw population features have lower discriminative power once spatial access is accounted for.
        """)
