import os
import numpy as np
import pandas as pd
import joblib
from src.shipment_prep import clean_shipments, build_features

# 1. Load Data
new_ship = pd.read_csv("data/new_shipments.csv")
hubs_df = pd.read_csv("data/hubs.csv")

# Load Models/Configs
final_pipe = joblib.load("models/delay_model.pkl")
threshold = float(pd.read_csv("models/delay_threshold.csv").iloc[0]["threshold"])
features_list = pd.read_csv("models/feature_order.csv")["feature"].tolist()
impute_df = pd.read_csv("models/impute_values.csv").set_index("column")

medians = {
    "distance_km": impute_df.loc["distance_km", "median"],
    "parcel_weight_kg": impute_df.loc["parcel_weight_kg", "median"],
    "declared_value_inr": impute_df.loc["declared_value_inr", "median"]
}

# 2. Preprocess & Feature Build
clean_df, _ = clean_shipments(new_ship, medians=medians)
featured_df = build_features(clean_df, hubs_df)

# Arrange Columns in exact feature order
X_new = featured_df[features_list]

# 3. Predict & Action Routing
probs = final_pipe.predict_proba(X_new)[:, 1]
featured_df["delay_probability"] = np.round(probs, 4)
featured_df["flagged"] = (featured_df["delay_probability"] >= threshold).astype(int)

# Load hub segments
hub_segs = pd.read_csv("outputs/hub_segments.csv").set_index("hub_id")["segment"].to_dict()
featured_df["segment"] = featured_df["hub_id"].map(hub_segs)

def assign_action(row):
    if row["flagged"] == 1:
        if row["segment"] == "Escalate":
            return "Escalate to hub manager"
        else:
            return "Proactive customer call"
    return "No action"

featured_df["action"] = featured_df.apply(assign_action, axis=1)

# 4. Save Final CSV
out_cols = ["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]
output_df = featured_df[out_cols]
os.makedirs("outputs", exist_ok=True)
output_df.to_csv("outputs/new_shipment_actions.csv", index=False)

print("Part D Scoring complete.")
