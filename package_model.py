import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

from src.shipment_prep import FEATURES, clean_shipments


os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# --------------------------------------------------
# 1. Read required files
# --------------------------------------------------
df = pd.read_csv("data/modelling_table.csv")
raw = pd.read_csv("data/raw_shipments.csv")

config = pd.read_csv(
    "outputs/part_c_final_config.csv"
).iloc[0]


# --------------------------------------------------
# 2. Rebuild exact Part C split
# --------------------------------------------------
X = df[FEATURES]
y = df["is_delayed"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)


# --------------------------------------------------
# 3. Read chosen configuration
# --------------------------------------------------
selector_k = int(config["selector_k"])
n_estimators = int(config["n_estimators"])

max_depth_raw = config["max_depth"]

if str(max_depth_raw) == "unlimited":
    max_depth = None
else:
    max_depth = int(max_depth_raw)

min_samples_leaf = int(
    config["min_samples_leaf"]
)

threshold = float(
    config["threshold"]
)


# --------------------------------------------------
# 4. Build final pipeline
# --------------------------------------------------
final_pipeline = Pipeline(
    [
        (
            "selector",
            SelectKBest(
                score_func=f_classif,
                k=selector_k,
            ),
        ),
        (
            "model",
            RandomForestClassifier(
                random_state=42,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
            ),
        ),
    ]
)


# --------------------------------------------------
# 5. Fit on training rows only
# --------------------------------------------------
final_pipeline.fit(
    X_train,
    y_train,
)


# --------------------------------------------------
# 6. Save model
# --------------------------------------------------
joblib.dump(
    final_pipeline,
    "models/delay_model.pkl",
)


# --------------------------------------------------
# 7. Save threshold
# --------------------------------------------------
pd.DataFrame(
    {
        "threshold": [threshold]
    }
).to_csv(
    "models/delay_threshold.csv",
    index=False,
)


# --------------------------------------------------
# 8. Save feature order
# --------------------------------------------------
pd.DataFrame(
    {
        "feature": FEATURES
    }
).to_csv(
    "models/feature_order.csv",
    index=False,
)


# --------------------------------------------------
# 9. Recompute and save imputation medians
# --------------------------------------------------
_, medians = clean_shipments(
    raw,
    medians=None,
)

impute_values = pd.DataFrame(
    [
        {
            "column": column,
            "median": value,
        }
        for column, value in medians.items()
    ]
)

impute_values["median"] = (
    impute_values["median"].round(4)
)

impute_values.to_csv(
    "models/impute_values.csv",
    index=False,
)


# --------------------------------------------------
# 10. Reload model and verify probabilities
# --------------------------------------------------
reloaded_model = joblib.load(
    "models/delay_model.pkl"
)

in_memory_probability = (
    final_pipeline.predict_proba(
        X_test
    )[:, 1]
)

reloaded_probability = (
    reloaded_model.predict_proba(
        X_test
    )[:, 1]
)

max_probability_diff = float(
    np.max(
        np.abs(
            in_memory_probability
            - reloaded_probability
        )
    )
)


reload_check = pd.DataFrame(
    [
        {
            "metric": "test_rows_scored",
            "value": len(X_test),
        },
        {
            "metric": "reload_max_abs_probability_diff",
            "value": round(
                max_probability_diff,
                4,
            ),
        },
        {
            "metric": "threshold",
            "value": round(
                threshold,
                4,
            ),
        },
    ]
)

reload_check.to_csv(
    "outputs/part_d_reload_check.csv",
    index=False,
)


# --------------------------------------------------
# 11. Console summary
# --------------------------------------------------
print("Model packaging complete")

print("Test rows scored:", len(X_test))

print(
    "Reload max probability difference:",
    max_probability_diff,
)

print(
    "Threshold:",
    threshold,
)

print("\nFiles written:")
print("models/delay_model.pkl")
print("models/delay_threshold.csv")
print("models/feature_order.csv")
print("models/impute_values.csv")
print("outputs/part_d_reload_check.csv")
