"""
Cluster Analysis page — ML insights, feature scatter plots, silhouette scores.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.clustering import CLUSTER_COLORS, TIER_ORDER
from utils.shared_data import get_pipeline_data


def render():
    st.markdown("## Cluster Analysis")
    st.caption("Agglomerative (Ward) clustering with k=3: High Pressure / Emerging Pressure / Well-Served.")

    data = get_pipeline_data()
    clustered, silhouettes = data["clustered"], data["silhouettes"]

    # ── Silhouette score chart ────────────────────────────────────────────────
    col_sil, col_info = st.columns([1, 2])

    with col_sil:
        st.subheader("Optimal k (Silhouette)")
        sil_df = pd.DataFrame(
            list(silhouettes.items()), columns=["k", "Silhouette Score"]
        )
        fig_sil = px.line(
            sil_df, x="k", y="Silhouette Score",
            markers=True,
            color_discrete_sequence=["#764ba2"],
            title=f"Silhouette Score by k (k=3 fixed)",
        )
        # Mark k=3
        fig_sil.add_vline(x=3, line_dash="dash", line_color="#D62728",
                          annotation_text="k=3 (High Pressure / Emerging / Well-Served)", annotation_position="top")
        fig_sil.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            font_family="Arial", height=280,
            xaxis=dict(showgrid=False, dtick=1),
            yaxis=dict(gridcolor="#F0F0F0"),
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    with col_info:
        st.subheader("Cluster Centroids")
        centroid_data = (
            clustered.groupby("cluster_label")[
                ["senior_pop_2025", "senior_growth_rate", "infra_density", "dist_nearest_hospital_km", "dist_nearest_poly_km"]
            ].mean().reset_index()
        )
        # Sort by defined tier order; any unexpected labels go to the end
        centroid_data["_order"] = centroid_data["cluster_label"].apply(
            lambda x: TIER_ORDER.index(x) if x in TIER_ORDER else 99
        )
        centroid_data = centroid_data.sort_values("_order").drop(columns="_order")
        centroid_data["senior_pop_2025"]    = centroid_data["senior_pop_2025"].map("{:,.0f}".format)
        centroid_data["senior_growth_rate"] = centroid_data["senior_growth_rate"].map("{:+.1%}".format)
        centroid_data["infra_density"]      = centroid_data["infra_density"].map("{:.2f}".format)
        centroid_data["dist_nearest_hospital_km"] = centroid_data["dist_nearest_hospital_km"].map("{:.1f} km".format)
        centroid_data["dist_nearest_poly_km"]     = centroid_data["dist_nearest_poly_km"].map("{:.1f} km".format)
        centroid_data.columns = ["Cluster", "Avg No. of Elderly", "Avg Growth Rate", "Avg Infra Density", "Avg Dist Hospital", "Avg Dist Polyclinic"]
        st.dataframe(centroid_data, use_container_width=True, hide_index=True)

        st.markdown("""
> **Cluster interpretation:**
> - **High Pressure** — High aging index, high growth rate, low infrastructure — immediate planning priority
> - **Emerging Pressure** — Moderate aging with accelerating growth — monitor closely, invest proactively
> - **Well-Served** — Low demand pressure or strong infrastructure coverage — maintain current provision
        """)

    st.divider()

    # ── 2D Scatter: Demand-Supply Ratio vs Infra Density ─────────────────────
    st.subheader("Feature Space: Demand-Supply Ratio vs Infrastructure Density")
    fig_scatter = px.scatter(
        clustered.dropna(subset=["demand_supply_ratio", "infra_density"]),
        x="demand_supply_ratio",
        y="infra_density",
        color="cluster_label",
        color_discrete_map=CLUSTER_COLORS,
        category_orders={"cluster_label": TIER_ORDER},
        size="senior_pop_2025",
        size_max=25,
        hover_name="SZ",
        hover_data={"PA": True, "infra_density": ":.3f",
                    "senior_pop_2025": ":,", "senior_growth_rate": ":.1%",
                    "demand_supply_ratio": ":.2f"},
        title="Subzones by Demand-Supply Ratio vs Infrastructure Density (bubble = elderly population)",
        labels={
            "demand_supply_ratio": "Demand-Supply Ratio",
            "infra_density": "Infrastructure Density",
            "cluster_label": "Risk Cluster",
        },
        opacity=0.8,
    )
    fig_scatter.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="Arial", height=480,
        xaxis=dict(showgrid=True, gridcolor="#F0F0F0"),
        yaxis=dict(showgrid=True, gridcolor="#F0F0F0"),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── 3D Scatter ────────────────────────────────────────────────────────────
    st.subheader("3D Feature Space")
    df3d = clustered.dropna(subset=["demand_supply_ratio", "senior_pop_2025", "infra_density", "senior_growth_rate"])
    fig3d = px.scatter_3d(
        df3d,
        x="demand_supply_ratio",
        y="senior_pop_2025",
        z="infra_density",
        color="cluster_label",
        color_discrete_map=CLUSTER_COLORS,
        category_orders={"cluster_label": TIER_ORDER},
        hover_name="SZ",
        hover_data={"PA": True, "senior_growth_rate": ":.1%"},
        title="3D Cluster View: Demand-Supply Ratio × Elderly Population × Infrastructure Density",
        labels={
            "demand_supply_ratio": "Demand-Supply Ratio",
            "senior_pop_2025": "No. of Elderly",
            "infra_density": "Infra Density",
            "cluster_label": "Cluster",
        },
        opacity=0.75,
        size_max=6,
    )
    fig3d.update_layout(
        font_family="Arial",
        scene=dict(
            xaxis_title="Demand-Supply Ratio",
            yaxis_title="No. of Elderly",
            zaxis_title="Infra Density",
        ),
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=-0.12),
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # ── Raw feature table ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Full Feature Table")
    with st.expander("Show all subzone features and cluster assignments"):
        display = clustered[[
            "PA", "SZ", "cluster_label",
            "senior_pop_2025", "senior_growth_rate",
            "total_pop_2025", "infra_density",
            "dist_nearest_hospital_km", "dist_nearest_poly_km"
        ]].copy()
        display["senior_pop_2025"]          = display["senior_pop_2025"].map("{:,.0f}".format)
        display["senior_growth_rate"]       = display["senior_growth_rate"].map("{:+.2%}".format)
        display["total_pop_2025"]           = display["total_pop_2025"].map("{:,.0f}".format)
        display["infra_density"]            = display["infra_density"].map("{:.2f}".format)
        display["dist_nearest_hospital_km"] = display["dist_nearest_hospital_km"].map("{:.1f}".format)
        display["dist_nearest_poly_km"]     = display["dist_nearest_poly_km"].map("{:.1f}".format)
        display.columns = [
            "Planning Area", "Subzone", "Risk Cluster",
            "No. of Elderly", "Growth Rate",
            "Total Pop", "Infra Density",
            "Dist Hospital (km)", "Dist Polyclinic (km)"
        ]
        st.dataframe(display, use_container_width=True, hide_index=True)
