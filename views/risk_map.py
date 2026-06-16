"""
Risk Map — clusters + infrastructure with distinct SVG icons.
"""
import streamlit as st
import folium
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.shared_data import get_pipeline_data


# ── SVG DivIcons ──────────────────────────────────────────────────────────────

def _icon_eldercare() -> folium.DivIcon:
    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"
        fill="#2471A3" stroke="white" stroke-width="1.5">
      <path d="M3 9.5L12 3l9 6.5V21H3V9.5z"/>
      <rect x="9" y="14" width="6" height="7" fill="white" stroke="#2471A3" stroke-width="1"/>
    </svg>"""
    return folium.DivIcon(html=svg, icon_size=(18, 18), icon_anchor=(9, 9))


def _icon_dementia() -> folium.DivIcon:
    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"
        fill="#7D3C98" stroke="white" stroke-width="1">
      <path d="M12 21s-9-5.5-9-12a5 5 0 0 1 9-3 5 5 0 0 1 9 3c0 6.5-9 12-9 12z"/>
    </svg>"""
    return folium.DivIcon(html=svg, icon_size=(18, 18), icon_anchor=(9, 9))


def _icon_polyclinic() -> folium.DivIcon:
    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24">
      <rect x="2" y="2" width="20" height="20" rx="4" fill="#1A7A3C" stroke="white" stroke-width="1"/>
      <rect x="10" y="5" width="4" height="14" fill="white"/>
      <rect x="5" y="10" width="14" height="4" fill="white"/>
    </svg>"""
    return folium.DivIcon(html=svg, icon_size=(18, 18), icon_anchor=(9, 9))


def _icon_hospital() -> folium.DivIcon:
    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24">
      <rect x="1" y="1" width="22" height="22" rx="4" fill="#C0392B" stroke="white" stroke-width="1.5"/>
      <text x="12" y="17" font-family="Arial" font-size="13" font-weight="bold"
            fill="white" text-anchor="middle">H</text>
    </svg>"""
    return folium.DivIcon(html=svg, icon_size=(20, 20), icon_anchor=(10, 10))


# ── GeoJSON loader (cached) ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_geojson_lookup() -> dict:
    """Load URA GeoJSON → {(PA_UPPER, SZ_UPPER): geojson_feature}. Cached."""
    import json
    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data", "MasterPlan2019SubzoneBoundaryNoSeaGEOJSON.geojson",
    )
    with open(path) as f:
        gj = json.load(f)
    return {
        (feat["properties"]["PLN_AREA_N"].strip().upper(),
         feat["properties"]["SUBZONE_N"].strip().upper()): feat
        for feat in gj["features"]
    }


# ── Map builder ────────────────────────────────────────────────────────────────

def build_map(clustered: pd.DataFrame, ec: pd.DataFrame, dg: pd.DataFrame,
              poly: pd.DataFrame, hospitals: pd.DataFrame) -> folium.Map:
    """Build the Folium map with all layers as named FeatureGroups."""
    from utils.clustering import CLUSTER_COLORS, TIER_ORDER

    m = folium.Map(location=[1.3521, 103.8198], zoom_start=11, tiles=None)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr=" ", name="Base Map", show=True, control=False,
    ).add_to(m)

    gz_lookup = _load_geojson_lookup()

    # ── Cluster FeatureGroups ──────────────────────────────────────────────
    cluster_groups: dict[str, folium.FeatureGroup] = {}
    for label in TIER_ORDER:
        if label in clustered["cluster_label"].values:
            cluster_groups[label] = folium.FeatureGroup(
                name=label, show=True, control=True
            )

    for _, row in clustered.iterrows():
        label = row["cluster_label"]
        group = cluster_groups.get(label)
        if group is None:
            continue
        key  = (row["PA"].strip().upper(), row["SZ"].strip().upper())
        feat = gz_lookup.get(key)
        if feat is None:
            continue

        color = row["cluster_color"]
        popup_html = (
            f"<b>{row['SZ'].title()}</b><br>"
            f"<i>{row['PA'].title()}</i><br><br>"
            f"<b>Risk Tier:</b> {label}<br>"
            f"<b>Elderly (2025):</b> {int(row['senior_pop_2025']):,}<br>"
            f"<b>Growth Rate:</b> {row['senior_growth_rate']:+.1%}<br>"
            f"<b>Total Pop:</b> {int(row['total_pop_2025']):,}<br>"
            f"<b>Infra Density:</b> {row['infra_density']:.3f}"
        )
        folium.GeoJson(
            feat,
            style_function=lambda _, c=color: {
                "fillColor": c, "color": c,
                "weight": 1.2, "fillOpacity": 0.55, "opacity": 0.9,
            },
            highlight_function=lambda _: {
                "fillOpacity": 0.85, "weight": 2.5, "color": "#222222",
            },
            popup=folium.Popup(popup_html, max_width=240),
        ).add_to(group)

        # Centroid tooltip marker
        coords = feat["geometry"]["coordinates"]
        ring   = coords[0] if feat["geometry"]["type"] == "Polygon" else max(
            [r[0] for r in coords], key=len
        )
        c_lat = sum(c[1] for c in ring) / len(ring)
        c_lon = sum(c[0] for c in ring) / len(ring)
        folium.CircleMarker(
            location=[c_lat, c_lon], radius=1,
            color="rgba(0,0,0,0)", fill=False,
            tooltip=f"{row['SZ'].title()} | {label} | Elderly: {int(row['senior_pop_2025']):,}",
        ).add_to(group)

    for group in cluster_groups.values():
        group.add_to(m)

    # ── Infrastructure FeatureGroups ───────────────────────────────────────
    def _build_infra_group(df: pd.DataFrame, name: str, icon_fn) -> folium.FeatureGroup:
        group = folium.FeatureGroup(name=name, show=True, control=True)
        for _, r in df.dropna(subset=["lat", "lon"]).iterrows():
            folium.Marker(
                location=[r["lat"], r["lon"]],
                icon=icon_fn(),
                tooltip=r["name"],
                popup=folium.Popup(r["name"], max_width=180),
            ).add_to(group)
        return group

    _build_infra_group(ec,        "Eldercare Services",    _icon_eldercare).add_to(m)
    _build_infra_group(dg,        "Dementia-Friendly GTPs", _icon_dementia).add_to(m)
    _build_infra_group(poly,      "Polyclinics",            _icon_polyclinic).add_to(m)
    _build_infra_group(hospitals, "Public Hospitals",       _icon_hospital).add_to(m)

    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    return m


_MAP_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "pipeline_cache", "risk_map.html"
)


@st.cache_data(show_spinner=False)
def _get_map_html(n_clustered: int, n_ec: int, n_dg: int, n_poly: int, n_hosp: int,
                   clustered: pd.DataFrame, ec: pd.DataFrame, dg: pd.DataFrame,
                   poly: pd.DataFrame, hospitals: pd.DataFrame) -> str:
    """Build the map once and cache its rendered HTML (picklable, unlike folium.Map).

    The n_* args form the cache key; folium.Map itself can't be hashed/cached,
    but the resulting HTML string can — avoiding rebuild on every page revisit.
    Also persisted to disk so a cold server restart skips the GeoJSON join and
    folium build entirely.
    """
    if os.path.exists(_MAP_CACHE_PATH):
        with open(_MAP_CACHE_PATH, "r", encoding="utf-8") as f:
            return f.read()

    m = build_map(clustered, ec, dg, poly, hospitals)
    html = m._repr_html_()

    os.makedirs(os.path.dirname(_MAP_CACHE_PATH), exist_ok=True)
    with open(_MAP_CACHE_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    return html


def render():
    from utils.clustering import CLUSTER_COLORS, TIER_ORDER

    st.markdown("## Subzone Risk Map")
    st.caption("Polygon fill = Risk Cluster  |  Click a subzone for details  |  Toggle layers via the control (top-right of map)")

    data = get_pipeline_data()
    clustered, ec, dg, poly, hospitals = (
        data["clustered"], data["ec"], data["dg"], data["poly"], data["hospitals"]
    )

    # ── Legend ────────────────────────────────────────────────────────────
    active_clusters = [t for t in TIER_ORDER if t in clustered["cluster_label"].values]

    # Inline SVGs matching the map's DivIcons exactly
    _svg_eldercare = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" '
        'fill="#2471A3" stroke="white" stroke-width="1.5">'
        '<path d="M3 9.5L12 3l9 6.5V21H3V9.5z"/>'
        '<rect x="9" y="14" width="6" height="7" fill="white" stroke="#2471A3" stroke-width="1"/>'
        '</svg>'
    )
    _svg_dementia = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" '
        'fill="#7D3C98" stroke="white" stroke-width="1">'
        '<path d="M12 21s-9-5.5-9-12a5 5 0 0 1 9-3 5 5 0 0 1 9 3c0 6.5-9 12-9 12z"/>'
        '</svg>'
    )
    _svg_polyclinic = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24">'
        '<rect x="2" y="2" width="20" height="20" rx="4" fill="#1A7A3C" stroke="white" stroke-width="1"/>'
        '<rect x="10" y="5" width="4" height="14" fill="white"/>'
        '<rect x="5" y="10" width="14" height="4" fill="white"/>'
        '</svg>'
    )
    _svg_hospital = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24">'
        '<rect x="1" y="1" width="22" height="22" rx="4" fill="#C0392B" stroke="white" stroke-width="1.5"/>'
        '<text x="12" y="17" font-family="Arial" font-size="13" font-weight="bold" '
        'fill="white" text-anchor="middle">H</text>'
        '</svg>'
    )

    infra_items = [
        (_svg_eldercare,  "Eldercare Services"),
        (_svg_dementia,   "Dementia-Friendly GTPs"),
        (_svg_polyclinic, "Polyclinics"),
        (_svg_hospital,   "Public Hospitals"),
    ]

    st.markdown("**Legend**")
    leg_cols = st.columns(len(active_clusters) + len(infra_items))
    for col, label in zip(leg_cols, active_clusters):
        color = CLUSTER_COLORS.get(label, "#95A5A6")
        col.markdown(
            f'<div style="background:{color};border-radius:6px;padding:5px 8px;'
            f'color:white;font-size:0.75rem;font-weight:600;text-align:center;">'
            f'<span style="display:inline-block;width:10px;height:10px;background:{color};'
            f'border:1px solid white;border-radius:2px;vertical-align:middle;margin-right:5px;"></span>'
            f'{label}</div>', unsafe_allow_html=True,
        )
    for col, (svg, label) in zip(leg_cols[len(active_clusters):], infra_items):
        col.markdown(
            f'<div style="background:#F8F9FA;border:1px solid #DEE2E6;'
            f'border-radius:4px;padding:5px 8px;font-size:0.75rem;text-align:center;'
            f'display:flex;align-items:center;justify-content:center;gap:5px;">'
            f'{svg}<span>{label}</span></div>', unsafe_allow_html=True,
        )

    st.divider()

    # ── Map — render cached HTML directly (avoids st_folium remount bugs) ──
    map_html = _get_map_html(
        len(clustered), len(ec), len(dg), len(poly), len(hospitals),
        clustered, ec, dg, poly, hospitals,
    )
    st.components.v1.html(map_html, height=600, scrolling=False)

    # ── Data table ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Subzone Data Explorer")

    display = clustered[
        ["PA", "SZ", "cluster_label", "senior_pop_2025",
         "senior_growth_rate", "total_pop_2025", "infra_density"]
    ].copy()
    display.columns = ["Planning Area", "Subzone", "Risk Tier",
                       "Elderly (2025)", "Growth Rate", "Total Pop", "Infra Density"]

    f1, f2, f3 = st.columns([2, 2, 1.5])
    with f1:
        search_text = st.text_input("Search area or subzone", placeholder="e.g. Tampines, Woodlands…")
    with f2:
        cluster_filter = st.multiselect(
            "Filter by risk tier", options=sorted(display["Risk Tier"].unique()),
            default=sorted(display["Risk Tier"].unique()), key="table_cluster_filter",
        )
    with f3:
        sort_col = st.selectbox("Sort by", ["Elderly (2025)", "Growth Rate", "Total Pop", "Infra Density"])

    mask = display["Risk Tier"].isin(cluster_filter)
    if search_text:
        q = search_text.lower()
        mask &= (
            display["Planning Area"].str.lower().str.contains(q, na=False) |
            display["Subzone"].str.lower().str.contains(q, na=False)
        )
    filtered = display[mask].copy().sort_values(sort_col, ascending=False)
    filtered["Elderly (2025)"]  = filtered["Elderly (2025)"].map("{:,.0f}".format)
    filtered["Growth Rate"]     = filtered["Growth Rate"].map("{:+.1%}".format)
    filtered["Total Pop"]       = filtered["Total Pop"].map("{:,.0f}".format)
    filtered["Infra Density"]   = filtered["Infra Density"].map("{:.3f}".format)

    st.caption(f"Showing **{len(filtered)}** of **{len(display)}** subzones")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
