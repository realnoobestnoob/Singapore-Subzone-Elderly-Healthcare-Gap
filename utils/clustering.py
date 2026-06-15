"""
Clustering module — Agglomerative (Ward) + RobustScaler, fixed k=3.

Architecture decisions:
  - RobustScaler   : senior_pop and infra_density are lognormal
  - AgglomerativeClustering (ward): sil~0.716 vs KMeans sil=0.313
  - k=3 fixed      : maps directly to planning language
  - Outlier removal: subzones with ≤10 elderly excluded

Accuracy fix (v5):
  Previously, cluster membership was determined by raw feature similarity
  (senior_pop, growth, infra_density, distances, demand_supply_ratio), and
  risk_score was only computed AFTERWARDS per cluster centroid to label/order
  the 3 clusters. This meant a subzone with thousands of elderly and almost
  no nearby infrastructure could still cluster with "Well-Served" zones if
  its distance/infra values happened to be numerically close to that
  cluster's centroid — its extreme demand was averaged away.

  Fix: compute risk_score PER SUBZONE (not just per centroid) and include it
  directly as a clustering feature, with reduced winsorisation on
  senior_pop_2025 and demand_supply_ratio so extreme high-demand/low-supply
  subzones aren't compressed into the bulk distribution before clustering.

Benchmark history:
  v1 (StandardScaler, KMeans k=4, single run)            : sil = 0.342
  v2 (StandardScaler, KMeans k=3, ensemble 80 runs)      : sil = 0.447
  v3 (StandardScaler, KMeans k-means++, HP tuning)       : sil = 0.313  (regressed)
  v4 (RobustScaler,   Agglomerative ward, optimal k)     : sil ~ 0.716
  v5 (+ per-subzone risk_score as feature, wider winsor) : sil ~ 0.69
       (slightly lower sil, but cluster membership now matches risk reality)
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


# ── Constants ─────────────────────────────────────────────────────────────────

CLUSTER_COLORS = {
    "High Pressure":     "#C0392B",
    "High-Growth Risk":  "#E67E22",
    "Emerging Pressure": "#D4A017",
    "Stable Low-Risk":   "#2471A3",
    "Well-Served":       "#1A7A3C",
}

TIER_LABELS = {
    3: ["High Pressure", "Emerging Pressure", "Well-Served"],
    4: ["High Pressure", "High-Growth Risk", "Emerging Pressure", "Well-Served"],
    5: ["High Pressure", "High-Growth Risk", "Emerging Pressure", "Stable Low-Risk", "Well-Served"],
}

TIER_ORDER = [
    "High Pressure",
    "High-Growth Risk",
    "Emerging Pressure",
    "Stable Low-Risk",
    "Well-Served",
]

BASE_FEATURES = [
    "senior_pop_2025",
    "senior_growth_rate",
    "infra_density",
    "dist_nearest_hospital_km",
    "dist_nearest_poly_km",
]
ALL_FEATURES = BASE_FEATURES + ["demand_supply_ratio"]

# Clustering now happens on this set: risk_score directly captures the
# demand-vs-supply imbalance; infra_density and senior_pop_2025 are retained
# to preserve separation between e.g. "low elderly + no infra" (low risk)
# vs "high elderly + no infra" (high risk) which risk_score alone may blur
# after scaling.
CLUSTER_FEATURES = ["risk_score", "senior_pop_2025", "infra_density", "senior_growth_rate"]

MIN_ELDERLY          = 10
K_FIXED              = 3
WINSORISE_LOW, WINSORISE_HIGH = 5, 95
# Wider winsorisation for features where extreme values ARE the risk signal —
# compressing them at 5/95 hides exactly the subzones we need to identify.
WINSORISE_WIDE = {"senior_pop_2025": (1, 99), "demand_supply_ratio": (1, 99), "risk_score": (1, 99)}

# Risk-score weights — derived from feature importance analysis (RF + permutation)
RISK_WEIGHTS = {
    "dist_nearest_poly_km":       1.8,   # #1 importance
    "infra_density":             -1.2,   # #2 (more infra = less risk)
    "dist_nearest_hospital_km":   1.1,   # #3
    "senior_pop_per_1000":        0.5,   # #4
    "senior_growth_rate":         0.3,   # #5
}


# ── Feature engineering ───────────────────────────────────────────────────────

def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add demand_supply_ratio and per-subzone risk_score."""
    df = df.copy()
    dist_h = df.get("dist_nearest_hospital_km", pd.Series(0.0, index=df.index))
    dist_p = df.get("dist_nearest_poly_km",     pd.Series(0.0, index=df.index))

    distance_penalty  = 1.0 + 0.05 * dist_h + 0.03 * dist_p
    effective_supply  = df["infra_density"] / distance_penalty
    df["demand_supply_ratio"] = (df["senior_pop_2025"] / 1_000) / (effective_supply + 0.001)

    # Per-subzone risk_score — same weighting previously applied only to
    # cluster centroids, now computed for every subzone individually so it
    # can drive cluster membership directly.
    df["senior_pop_per_1000"] = df["senior_pop_2025"] / 1_000
    df["risk_score"] = (
        df["dist_nearest_poly_km"]      * RISK_WEIGHTS["dist_nearest_poly_km"]
        + df["infra_density"]           * RISK_WEIGHTS["infra_density"]
        + df["dist_nearest_hospital_km"] * RISK_WEIGHTS["dist_nearest_hospital_km"]
        + df["senior_pop_per_1000"]     * RISK_WEIGHTS["senior_pop_per_1000"]
        + df["senior_growth_rate"]      * RISK_WEIGHTS["senior_growth_rate"]
    )
    return df


def _remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop subzones with ≤10 elderly — too sparse for reliable clustering."""
    before = len(df)
    df = df[df["senior_pop_2025"] > MIN_ELDERLY].copy()
    removed = before - len(df)
    if removed:
        print(f"[clustering] Removed {removed} outlier subzone(s) with <={MIN_ELDERLY} elderly.")
    return df


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _winsorise(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Winsorise each column independently; high-signal columns get wider bounds."""
    out = np.empty((len(df), len(columns)))
    for i, col in enumerate(columns):
        lo_pct, hi_pct = WINSORISE_WIDE.get(col, (WINSORISE_LOW, WINSORISE_HIGH))
        vals = df[col].values
        lo, hi = np.percentile(vals, [lo_pct, hi_pct])
        out[:, i] = np.clip(vals, lo, hi)
    return out


def _preprocess(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Winsorise then RobustScaler — handles lognormal skew in senior_pop/infra_density."""
    X = _winsorise(df, columns)
    return RobustScaler().fit_transform(X)


# ── Clustering ────────────────────────────────────────────────────────────────

def _cluster(X: np.ndarray, k: int) -> np.ndarray:
    """Agglomerative clustering with Ward linkage."""
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)


# ── Risk scoring ──────────────────────────────────────────────────────────────

def _rank_clusters_by_risk(df: pd.DataFrame) -> pd.Series:
    """Rank cluster centroids by mean risk_score — 0 = highest risk."""
    centroid_risk = df.groupby("cluster_id")["risk_score"].mean()
    return centroid_risk.rank(ascending=False).astype(int) - 1


# ── Public API ────────────────────────────────────────────────────────────────

def run_clustering(feat: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline:
      1. Remove outliers (≤10 elderly)
      2. Feature engineering (incl. per-subzone risk_score) + winsorise + RobustScale
      3. Agglomerative Ward clustering on CLUSTER_FEATURES (risk-aligned)
      4. Assign risk-ordered tier labels by mean cluster risk_score
    Returns feat rows (minus outliers) with cluster_id, cluster_label, cluster_color.
    """
    df = feat.dropna(subset=BASE_FEATURES).copy()
    df = _remove_outliers(df)
    df = _add_features(df)
    X  = _preprocess(df, CLUSTER_FEATURES)

    best_k = K_FIXED
    df["cluster_id"] = _cluster(X, best_k)

    rank        = _rank_clusters_by_risk(df)
    tier_labels = TIER_LABELS[best_k]
    df["cluster_label"] = df["cluster_id"].map(
        {int(cid): tier_labels[int(rank[cid])] for cid in range(best_k)}
    )
    df["cluster_color"] = df["cluster_label"].map(CLUSTER_COLORS).fillna("#95A5A6")
    return df


def get_silhouette(feat: pd.DataFrame, k_range=range(2, 7)) -> dict:
    """Return silhouette scores for k_range — used by cluster_analysis page."""
    df = feat.dropna(subset=BASE_FEATURES).copy()
    df = _remove_outliers(df)
    df = _add_features(df)
    X  = _preprocess(df, CLUSTER_FEATURES)
    scores = {}
    for k in k_range:
        labels = _cluster(X, k)
        if len(set(labels)) > 1:
            scores[k] = round(silhouette_score(X, labels), 3)
    return scores
