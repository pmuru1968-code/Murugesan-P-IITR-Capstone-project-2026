import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# 1. Load Data
modelling_df = pd.read_csv("data/modelling_table.csv")

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value",
    "n_stops", "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity", "label_print_batch", "scanner_build_no", 
    "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]
TARGET = "is_delayed"

X = modelling_df[FEATURES]
y = modelling_df[TARGET]

# 2. Stratified Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

os.makedirs("outputs", exist_ok=True)

# Helper for metrics
def compute_row_metrics(model_name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    # Write precision as 0 for baseline to avoid warning or if no positive predictions
    if model_name == "blind_all_on_time":
        precision = 0.0
    else:
        precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "model": model_name,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1
    }

# 3. Blind Baseline & Candidates
models_metrics = []

# Blind Baseline
blind_preds = np.zeros(len(y_val))
models_metrics.append(compute_row_metrics("blind_all_on_time", y_val, blind_preds))

# Logistic Regression Pipeline
lr_pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
lr_pipe.fit(X_train, y_train)
models_metrics.append(compute_row_metrics("logistic_regression", y_val, lr_pipe.predict(X_val)))

# Choose Decision Tree depth by highest validation F1
best_dt_f1 = -1
best_dt_depth = 2
best_dt_preds = None

for d in [2, 3, 4, 5, 6, 8, 10]:
    dt = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    preds = dt.predict(X_val)
    f1 = f1_score(y_val, preds, zero_division=0)
    if f1 > best_dt_f1:
        best_dt_f1 = f1
        best_dt_depth = d
        best_dt_preds = preds

models_metrics.append(compute_row_metrics(f"decision_tree_depth_{best_dt_depth}", y_val, best_dt_preds))

# Random Forest Default
rf_default = RandomForestClassifier(n_estimators=100, random_state=42)
rf_default.fit(X_train, y_train)
models_metrics.append(compute_row_metrics("random_forest_default", y_val, rf_default.predict(X_val)))

models_df = pd.DataFrame(models_metrics)
models_df.to_csv("outputs/part_c_models.csv", index=False)

# 4. Feature Selection Signal Analysis
# Fit default forest to get feature importance
rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
rf_all.fit(X_train, y_train)
importances = rf_all.feature_importances_

# SelectKBest with f_classif
selector_all = SelectKBest(f_classif, k="all")
selector_all.fit(X_train, y_train)
f_scores = selector_all.scores_

feature_scores = pd.DataFrame({
    "feature": FEATURES,
    "f_score": f_scores,
    "rf_importance": importances
})
feature_scores["f_rank"] = feature_scores["f_score"].rank(ascending=False, method="min").astype(int)
feature_scores["is_noise"] = feature_scores["feature"].isin(NOISE_COLUMNS)
feature_scores.sort_values("f_rank", inplace=True)
feature_scores.to_csv("outputs/part_c_feature_scores.csv", index=False)

# F1 values for different k values
k_results = []
for k_val in [5, 7, 10, 12, 17]:
    k_pipe = Pipeline([
        ("selector", SelectKBest(f_classif, k=k_val)),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    k_pipe.fit(X_train, y_train)
    k_preds = k_pipe.predict(X_val)
    val_f1 = f1_score(y_val, k_preds, zero_division=0)
    k_results.append({"k": k_val, "val_f1": val_f1})

pd.DataFrame(k_results).to_csv("outputs/part_c_selectk.csv", index=False)

# 5. Tuning (using k=7 from here onwards)
selector_7 = SelectKBest(f_classif, k=7)
X_train_7 = selector_7.fit_transform(X_train, y_train)
X_val_7 = selector_7.transform(X_val)

# Hyperparameter grid
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10]
}

# Default forest (same 7 features)
rf_def_7 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_def_7.fit(X_train_7, y_train)
rf_def_7_preds = rf_def_7.predict(X_val_7)
val_f1_def_7 = f1_score(y_val, rf_def_7_preds, zero_division=0)

# Grid Search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5
)
grid_search.fit(X_train_7, y_train)
grid_preds = grid_search.predict(X_val_7)
val_f1_grid = f1_score(y_val, grid_preds, zero_division=0)

# Randomized Search
rand_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    n_iter=9,
    cv=5,
    random_state=42
)
rand_search.fit(X_train_7, y_train)
rand_preds = rand_search.predict(X_val_7)
val_f1_rand = f1_score(y_val, rand_preds, zero_division=0)

tuning_rows = [
    {
        "search": "grid_search",
        "candidates": 27,
        "cv_folds": 5,
        "total_fits": 135,
        "best_params": str(grid_search.best_params_),
        "best_cv_accuracy": grid_search.best_score_,
        "val_f1": val_f1_grid
    },
    {
        "search": "randomized_search",
        "candidates": 9,
        "cv_folds": 5,
        "total_fits": 45,
        "best_params": str(rand_search.best_params_),
        "best_cv_accuracy": rand_search.best_score_,
        "val_f1": val_f1_rand
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": "defaults",
        "best_cv_accuracy": np.nan,
        "val_f1": val_f1_def_7
    }
]
tuning_df = pd.DataFrame(tuning_rows)
tuning_df.to_csv("outputs/part_c_tuning.csv", index=False)

# 6. Choose Final Configuration
tuning_df_sorted = tuning_df.sort_values("val_f1", ascending=False)
best_tuning_row = tuning_df_sorted.iloc[0]
winner_candidate = best_tuning_row["search"]

if winner_candidate == "grid_search":
    final_params = grid_search.best_params_
elif winner_candidate == "randomized_search":
    final_params = rand_search.best_params_
else:
    final_params = {}

final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", RandomForestClassifier(random_state=42, **final_params))
])
final_model.fit(X_train, y_train)

# 7. Alert Threshold Tuning
val_probs = final_model.predict_proba(X_val)[:, 1]
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

threshold_rows = []
chosen_threshold = 0.1
found_threshold = False
max_t_for_recall = -1

for t in thresholds:
    t_preds = (val_probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, t_preds).ravel()
    accuracy = accuracy_score(y_val, t_preds)
    precision = precision_score(y_val, t_preds, zero_division=0)
    recall = recall_score(y_val, t_preds, zero_division=0)
    f1 = f1_score(y_val, t_preds, zero_division=0)
    
    threshold_rows.append({
        "threshold": t,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1
    })
    
    if recall >= 0.80:
        if t > max_t_for_recall:
            max_t_for_recall = t
            chosen_threshold = t
            found_threshold = True

if not found_threshold:
    chosen_threshold = 0.1

pd.DataFrame(threshold_rows).to_csv("outputs/part_c_thresholds.csv", index=False)

# 8. Test Evaluation at Chosen Threshold
test_probs = final_model.predict_proba(X_test)[:, 1]
test_preds_final = (test_probs >= chosen_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, test_preds_final).ravel()
accuracy = accuracy_score(y_test, test_preds_final)
precision = precision_score(y_test, test_preds_final, zero_division=0)
recall = recall_score(y_test, test_preds_final, zero_division=0)
f1 = f1_score(y_test, test_preds_final, zero_division=0)

final_test_row = {
    "threshold": chosen_threshold,
    "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1
}
pd.DataFrame([final_test_row]).to_csv("outputs/part_c_final_test.csv", index=False)

# Save Final Configuration file
config_row = {
    "selector_k": 7,
    "n_estimators": final_params.get("n_estimators", 100),
    "max_depth": final_params.get("max_depth", "unlimited"),
    "min_samples_leaf": final_params.get("min_samples_leaf", 1),
    "threshold": chosen_threshold,
    "chosen_from": winner_candidate
}
pd.DataFrame([config_row]).to_csv("outputs/part_c_final_config.csv", index=False)

print("Part C operations complete.")
