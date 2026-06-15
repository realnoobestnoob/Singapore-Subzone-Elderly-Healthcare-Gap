"""
Shared cached data pipeline — single source of truth for all pages.

All pages call get_pipeline_data() instead of running their own
load/build/cluster pipeline. Because @st.cache_data caches per-function,
defining this ONCE here (rather than duplicating get_data() in every page)
ensures the full pipeline runs exactly once per app session, regardless of
how many pages reference it. app.py warms this cache at startup so page
switches never trigger a recompute.

Results are ALSO persisted to disk (data/pipeline_cache/) so a cold server
restart skips ETL + clustering entirely — only the first-ever run (or after
deleting the cache) pays the full cost.
"""
import streamlit as st
import pandas as pd
import json
import os

from utils.data_pipeline import (
    load_population, build_features, load_eldercare, load_dementia_gtp,
    assign_infrastructure_to_subzones, load_clinics,
)
from utils.clustering import run_clustering, get_silhouette

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "pipeline_cache")
st.caption(f"Loaded clustered.csv: rows={len(clustered)}, cache_exists={os.path.exists(cache_path)}")

_DF_FILES = {
    "pop":        "pop.csv",
    "clustered":  "clustered.csv",
    "ec":         "ec.csv",
    "dg":         "dg.csv",
    "poly":       "poly.csv",
    "hospitals":  "hospitals.csv",
}
_SIL_FILE = "silhouettes.json"


def _cache_complete() -> bool:
    if not os.path.isdir(CACHE_DIR):
        return False
    paths = [os.path.join(CACHE_DIR, f) for f in _DF_FILES.values()] + \
            [os.path.join(CACHE_DIR, _SIL_FILE)]
    return all(os.path.exists(p) for p in paths)


def _load_from_disk() -> dict:
    data = {k: pd.read_csv(os.path.join(CACHE_DIR, f)) for k, f in _DF_FILES.items()}
    with open(os.path.join(CACHE_DIR, _SIL_FILE)) as fh:
        # JSON keys are strings — silhouette dict is keyed by int k
        data["silhouettes"] = {int(k): v for k, v in json.load(fh).items()}
    return data


def _save_to_disk(data: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    for k, fname in _DF_FILES.items():
        data[k].to_csv(os.path.join(CACHE_DIR, fname), index=False)
    with open(os.path.join(CACHE_DIR, _SIL_FILE), "w") as fh:
        json.dump(data["silhouettes"], fh)


@st.cache_data(show_spinner="Loading datasets and running clustering…")
def get_pipeline_data() -> dict:
    """
    Run the full ETL + clustering pipeline once (or load from disk cache).

    Returns a dict with all artefacts needed by Overview, Risk Map, Cluster
    Analysis, and Trends pages:
      pop        : raw population long-format DataFrame
      clustered  : feature table with cluster_label / cluster_color
      ec, dg     : eldercare / dementia GTP location DataFrames
      poly, hospitals : polyclinic / hospital location DataFrames
      silhouettes: dict of k -> silhouette score

    To force a recompute (e.g. after new SingStat data is added), delete the
    data/pipeline_cache/ directory and restart the app.
    """
    if _cache_complete():
        return _load_from_disk()

    pop = load_population()
    feat = build_features(pop)
    ec = load_eldercare()
    dg = load_dementia_gtp()
    poly, hospitals = load_clinics()
    feat = assign_infrastructure_to_subzones(feat, ec, dg, poly, hospitals)
    clustered = run_clustering(feat)
    silhouettes = get_silhouette(feat)

    data = {
        "pop": pop,
        "clustered": clustered,
        "ec": ec,
        "dg": dg,
        "poly": poly,
        "hospitals": hospitals,
        "silhouettes": silhouettes,
    }
    _save_to_disk(data)
    return data
