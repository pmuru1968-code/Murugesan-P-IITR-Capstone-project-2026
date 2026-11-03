import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load Data
modelling_df = pd.read_csv("data/modelling_table.csv")

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value",
    "n_stops", "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity", "label_print_batch", "scanner_build_no", 
    "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"
]
TARGET = "delivery_hours"

X = modelling_df[FEATURES]
y = modelling_df[TARGET]

# 2. Split into Train, Val, Test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 3. Scaling without leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Calculate metrics for part_b_checks
train_scaled_max_abs_mean = np.max(np.abs(np.mean(X_train_scaled, axis=0)))
test_scaled_max_abs_mean = np.max(np.abs(np.mean(X_test_scaled, axis=0)))

checks_dict = {
    "train_rows": len(X_train),
    "val_rows": len(X_val),
    "test_rows": len(X_test),
    "train_scaled_max_abs_mean": train_scaled_max_abs_mean,
    "test_scaled_max_abs_mean": test_scaled_max_abs_mean
}
checks_df = pd.DataFrame(list(checks_dict.items()), columns=["metric", "value"])
os.makedirs("outputs", exist_ok=True)
checks_df.to_csv("outputs/part_b_checks.csv", index=False)

# 4. Tree-depth diagnostic
depths = [1, 2, 3, 4, 6, 8, 12, None]
tree_diagnostic = []

for d in depths:
    model = DecisionTreeRegressor(max_depth=d, random_state=42)
    model.fit(X_train, y_train)
    
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    
    depth_label = "unlimited" if d is None else d
    tree_diagnostic.append({
        "max_depth": depth_label,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse
    })

tree_diagnostic_df = pd.DataFrame(tree_diagnostic)
tree_diagnostic_df.to_csv("outputs/part_b_tree_depth.csv", index=False)

# Choose best depth (ties -> smaller depth)
best_tree_row = tree_diagnostic_df.loc[tree_diagnostic_df["val_rmse"].idxmin()]
best_depth = best_tree_row["max_depth"]
if best_depth == "unlimited":
    best_depth_val = None
else:
    best_depth_val = int(best_depth)

# 5. Baseline & Four Candidates
metrics_list = []

def record_metrics(model_name, split, y_true, preds):
    mae = mean_absolute_error(y_true, preds)
    mse = mean_squared_error(y_true, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, preds)
    return {
        "model": model_name,
        "split": split,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

# Mean Baseline
baseline_preds = np.full(shape=len(y_val), fill_value=y_train.mean())
metrics_list.append(record_metrics("mean_baseline", "val", y_val, baseline_preds))

# Candidates dictionary
candidates = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth_val, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

for name, model in candidates.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    metrics_list.append(record_metrics(name, "val", y_val, preds))

# 6. Learning Rate Experiment
lr_list = [0.01, 0.1, 1.0]
lr_records = []
for lr in lr_list:
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
    gb.fit(X_train, y_train)
    preds = gb.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)
    lr_records.append({
        "learning_rate": lr,
        "val_rmse": rmse,
        "val_r2": r2
    })
lr_df = pd.DataFrame(lr_records)
lr_df.to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

# 7. CV Comparison and Final Model Selection
# Sort candidates by validation RMSE
val_metrics_df = pd.DataFrame(metrics_list)
candidate_ranks = val_metrics_df[val_metrics_df["model"].isin(candidates.keys())].sort_values("RMSE")
best_two = candidate_ranks["model"].head(2).tolist()

# Create combined train + validation pool for 5-fold CV
X_train_val = pd.concat([X_train, X_val], axis=0)
y_train_val = pd.concat([y_train, y_val], axis=0)

shared_kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_records = []

for model_name in best_two:
    model_obj = candidates[model_name]
    scores = cross_val_score(model_obj, X_train_val, y_train_val, cv=shared_kf, scoring="r2")
    cv_records.append({
        "model": model_name,
        "mean_r2": scores.mean(),
        "std_r2": scores.std()
    })

cv_df = pd.DataFrame(cv_records)
cv_df.to_csv("outputs/part_b_cv.csv", index=False)

# Winner is the one with the lowest validation RMSE
winner_name = candidate_ranks.iloc[0]["model"]
winner_model = candidates[winner_name]

# Score on Test
test_preds = winner_model.predict(X_test)
test_record = record_metrics(winner_name, "test", y_test, test_preds)
metrics_list.append(test_record)

# Save all model metrics
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv("outputs/part_b_metrics.csv", index=False)

print("Part B computations completed.")
