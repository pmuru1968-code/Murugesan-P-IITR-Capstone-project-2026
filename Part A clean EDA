import numpy as np
import pandas as pd
import os
from src.shipment_prep import clean_shipments, build_features

# 1. Read Data
raw_ship = pd.read_csv("data/raw_shipments.csv")
hubs_df = pd.read_csv("data/hubs.csv")

# --- Metrics before filling (post duplicate-removal) ---
df_no_dupes = raw_ship.drop_duplicates().copy()

service_type_spellings_before = df_no_dupes["service_type"].nunique()
missing_distance_before = df_no_dupes["distance_km"].isna().sum()
missing_payment_before = df_no_dupes["payment_mode"].isna().sum()
missing_value_before = df_no_dupes["declared_value"].isna().sum()

# Count sentinel rows (value > 100)
sentinel_weight_rows = df_no_dupes["parcel_weight_kg"] > 100
sentinel_count = sentinel_weight_rows.sum()

# Weight missingness check excluding sentinel rows
missing_weight_before = df_no_dupes.loc[~sentinel_weight_rows, "parcel_weight_kg"].isna().sum()

# 2. Run clean_shipments
clean_df, computed_medians = clean_shipments(raw_ship)

# --- Metrics after cleaning ---
service_type_spellings_after = clean_df["service_type"].nunique()

# 3. Build features
model_df = build_features(clean_df, hubs_df)

# Target column is_delayed
model_df["is_delayed"] = (model_df["delivery_hours"] > model_df["promised_hours"]).astype(int)

# Write modelling table
os.makedirs("data", exist_ok=True)
model_df.to_csv("data/modelling_table.csv", index=False)

# 4. Compute Metrics for Checks
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value",
    "n_stops", "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity", "label_print_batch", "scanner_build_no", 
    "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"
]

nulls_in_features_target = model_df[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()

dec_val_inr = model_df["declared_value_inr"]
log_dec_val = model_df["log_declared_value"]

# Outlier bounds for delivery hours
q25 = model_df["delivery_hours"].quantile(0.25)
q75 = model_df["delivery_hours"].quantile(0.75)
iqr = q75 - q25
lower_bound = q25 - 1.5 * iqr
upper_bound = q75 + 1.5 * iqr

outliers_count = ((model_df["delivery_hours"] < lower_bound) | (model_df["delivery_hours"] > upper_bound)).sum()

metrics_dict = {
    "rows_raw": len(raw_ship),
    "rows_no_duplicates": len(df_no_dupes),
    "service_type_spellings_before": service_type_spellings_before,
    "service_type_spellings_after": service_type_spellings_after,
    "missing_distance_before": missing_distance_before,
    "missing_parcel_weight_before": missing_weight_before,
    "missing_declared_value_before": missing_value_before,
    "missing_payment_mode_before": missing_payment_before,
    "sentinel_parcel_weight_rows": sentinel_count,
    "median_distance_km": computed_medians["distance_km"],
    "median_parcel_weight_kg": computed_medians["parcel_weight_kg"],
    "median_declared_value_inr": computed_medians["declared_value_inr"],
    "nulls_in_features_and_target": nulls_in_features_target,
    "declared_value_mean_inr": round(float(dec_val_inr.mean()), 2),
    "declared_value_median_inr": round(float(dec_val_inr.median()), 2),
    "log_declared_value_mean": round(float(log_dec_val.mean()), 4),
    "log_declared_value_median": round(float(log_dec_val.median()), 4),
    "hours_q25": round(float(q25), 2),
    "hours_q75": round(float(q75), 2),
    "hours_lower_bound": round(float(lower_bound), 2),
    "hours_upper_bound": round(float(upper_bound), 2),
    "hours_outliers": outliers_count,
    "delayed_parcels": int(model_df["is_delayed"].sum()),
    "overall_delay_rate": round(float(model_df["is_delayed"].mean()), 4)
}

checks_df = pd.DataFrame(list(metrics_dict.items()), columns=["metric", "value"])
os.makedirs("outputs", exist_ok=True)
checks_df.to_csv("outputs/part_a_checks.csv", index=False)

# 5. Delay Evidence Table
groups_list = []

# Service Type
for gp, sub in model_df.groupby("service_type"):
    groups_list.append({
        "dimension": "service_type",
        "group": gp,
        "shipments": len(sub),
        "delayed": sub["is_delayed"].sum(),
        "delay_rate": round(float(sub["is_delayed"].mean()), 4),
        "mean_hours": round(float(sub["delivery_hours"].mean()), 2)
    })

# Hub Tier
for gp, sub in model_df.groupby("hub_tier"):
    groups_list.append({
        "dimension": "hub_tier",
        "group": f"Tier {gp}",
        "shipments": len(sub),
        "delayed": sub["is_delayed"].sum(),
        "delay_rate": round(float(sub["is_delayed"].mean()), 4),
        "mean_hours": round(float(sub["delivery_hours"].mean()), 2)
    })

# Distance Band
bins = [0, 200, 400, 600, 2000]
labels = ["0-200", "201-400", "401-600", "601+"]
model_df["distance_band"] = pd.cut(model_df["distance_km"], bins=bins, labels=labels, right=True)

for gp, sub in model_df.groupby("distance_band", observed=False):
    groups_list.append({
        "dimension": "distance_band",
        "group": gp,
        "shipments": len(sub),
        "delayed": sub["is_delayed"].sum(),
        "delay_rate": round(float(sub["is_delayed"].mean()), 4),
        "mean_hours": round(float(sub["delivery_hours"].mean()), 2)
    })

delay_by_group_df = pd.DataFrame(groups_list)
delay_by_group_df.to_csv("outputs/part_a_delay_by_group.csv", index=False)

# 6. Correlations Table
numeric_cols = model_df.select_dtypes(include=[np.number]).columns.tolist()
# Exclude any targets or ID columns that shouldn't belong
# Keep index as first column as requested
corr_matrix = model_df[numeric_cols].corr()
corr_matrix.reset_index(names="feature").to_csv("outputs/part_a_correlations.csv", index=False)

print("Processing Complete. All files saved.")
