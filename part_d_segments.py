import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform

# 1. Load Data
modelling_df = pd.read_csv("data/modelling_table.csv")
hubs_df = pd.read_csv("data/hubs.csv")

# 2. Create Hub Profile
hub_profile = modelling_df.groupby("hub_id").agg(
    shipments=("shipment_id", "count"),
    mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"),
    mean_distance_km=("distance_km", "mean"),
    mean_n_stops=("n_stops", "mean")
).reset_index()

hub_profile = hub_profile.sort_values("hub_id").reset_index(drop=True)
os.makedirs("outputs", exist_ok=True)
hub_profile.to_csv("outputs/hub_profile.csv", index=False)

# 3. K-Means Segmentation Prep
clustering_cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
X_profile = hub_profile[clustering_cols].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_profile)

# Fit K=3
kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_3 = kmeans_3.fit_predict(X_scaled)

# 4. Separation Evidence
separation_records = []
for K in [2, 3, 4, 5]:
    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels_K = km.fit_predict(X_scaled)
    
    # Calculate across-over-within distance ratio
    dists = squareform(pdist(X_scaled, metric="euclidean"))
    n_hubs = len(X_scaled)
    
    within_dists = []
    across_dists = []
    
    for i in range(n_hubs):
        for j in range(i + 1, n_hubs):
            d_val = dists[i, j]
            if labels_K[i] == labels_K[j]:
                within_dists.append(d_val)
            else:
                across_dists.append(d_val)
                
    mean_within = np.mean(within_dists) if len(within_dists) > 0 else 0.0
    mean_across = np.mean(across_dists) if len(across_dists) > 0 else 0.0
    ratio = mean_across / mean_within if mean_within > 0 else 0.0
    
    # Cluster sizes descending
    sizes = pd.Series(labels_K).value_counts().sort_values(ascending=False).tolist()
    sizes_str = "-".join(map(str, sizes))
    
    separation_records.append({
        "K": K,
        "across_over_within": round(ratio, 4),
        "cluster_sizes": sizes_str
    })

pd.DataFrame(separation_records).to_csv("outputs/part_d_separation.csv", index=False)

# 5. Segment Names and Tier Check
# Calculate mean delay_rate for each of the 3 clusters to name them
hub_profile["cluster_temp"] = labels_3
cluster_delays = hub_profile.groupby("cluster_temp")["delay_rate"].mean().sort_values()

# Map lowest -> Standard, middle -> Watch, highest -> Escalate
name_mapping = {
    cluster_delays.index[0]: "Standard",
    cluster_delays.index[1]: "Watch",
    cluster_delays.index[2]: "Escalate"
}
hub_profile["segment"] = hub_profile["cluster_temp"].map(name_mapping)

# Merge hub info
hub_segments = hub_profile[["hub_id", "segment"]].merge(hubs_df[["hub_id", "hub_city", "hub_tier"]], on="hub_id")
hub_segments = hub_segments[["hub_id", "segment", "hub_city", "hub_tier"]]
hub_segments.to_csv("outputs/hub_segments.csv", index=False)

# Segment profile
segment_profile = hub_profile.groupby("segment").agg(
    hubs=("hub_id", "count"),
    mean_delivery_hours=("mean_delivery_hours", "mean"),
    delay_rate=("delay_rate", "mean"),
    mean_distance_km=("mean_distance_km", "mean"),
    mean_n_stops=("mean_n_stops", "mean")
).reset_index()

# Order segments logically
segment_order = ["Standard", "Watch", "Escalate"]
segment_profile["segment"] = pd.Categorical(segment_profile["segment"], categories=segment_order, ordered=True)
segment_profile = segment_profile.sort_values("segment").reset_index(drop=True)
segment_profile.to_csv("outputs/segment_profile.csv", index=False)

# Tier vs Segment Mismatch
tier_to_segment = {
    1: "Standard",
    2: "Watch",
    3: "Escalate"
}
hub_segments["implied_segment"] = hub_segments["hub_tier"].map(tier_to_segment)
mismatch = hub_segments[hub_segments["segment"] != hub_segments["implied_segment"]].drop(columns=["implied_segment"])
mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

print("Part D Segmentation operations complete.")
