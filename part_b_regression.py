import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.shipment_prep import FEATURES


os.makedirs("outputs", exist_ok=True)


# --------------------------------------------------
# 1. Read modelling table
# --------------------------------------------------
df = pd.read_csv("data/modelling_table.csv")

X = df[FEATURES]
y = df["delivery_hours"]


# --------------------------------------------------
# 2. Train / validation / test split
# --------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
)

print("Train rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))


# --------------------------------------------------
# 3. Scaling leakage check
# --------------------------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

train_scaled_max_abs_mean = float(
    np.abs(X_train_scaled.mean(axis=0)).max()
)

test_scaled_max_abs_mean = float(
    np.abs(X_test_scaled.mean(axis=0)).max()
)

checks = pd.DataFrame(
    [
        ("train_rows", len(X_train)),
        ("val_rows", len(X_val)),
        ("test_rows", len(X_test)),
        (
            "train_scaled_max_abs_mean",
            train_scaled_max_abs_mean,
        ),
        (
            "test_scaled_max_abs_mean",
            test_scaled_max_abs_mean,
        ),
    ],
    columns=["metric", "value"],
)

checks["value"] = checks["value"].apply(
    lambda x: round(float(x), 4)
    if isinstance(x, (float, np.floating))
    else x
)

checks.to_csv(
    "outputs/part_b_checks.csv",
    index=False,
)


# --------------------------------------------------
# Helper: calculate regression metrics
# --------------------------------------------------
def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }


metric_rows = []


# --------------------------------------------------
# 4. Mean baseline
# --------------------------------------------------
baseline_prediction = np.repeat(
    y_train.mean(),
    len(y_val),
)

baseline_metrics = regression_metrics(
    y_val,
    baseline_prediction,
)

metric_rows.append(
    {
        "model": "mean_baseline",
        "split": "val",
        **baseline_metrics,
    }
)


# --------------------------------------------------
# 5. Decision-tree depth diagnostic
# --------------------------------------------------
depths = [1, 2, 3, 4, 6, 8, 12, None]

tree_depth_rows = []

for depth in depths:
    tree = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42,
    )

    tree.fit(X_train, y_train)

    train_pred = tree.predict(X_train)
    val_pred = tree.predict(X_val)

    train_mse = mean_squared_error(
        y_train,
        train_pred,
    )

    val_mse = mean_squared_error(
        y_val,
        val_pred,
    )

    tree_depth_rows.append(
        {
            "max_depth": (
                "unlimited"
                if depth is None
                else str(depth)
            ),
            "train_rmse": np.sqrt(train_mse),
            "val_rmse": np.sqrt(val_mse),
            "_depth_value": depth,
        }
    )


tree_depth_df = pd.DataFrame(tree_depth_rows)

# Select lowest validation RMSE.
# If tied, choose the smaller finite depth.
best_tree_row = min(
    tree_depth_rows,
    key=lambda row: (
        row["val_rmse"],
        float("inf")
        if row["_depth_value"] is None
        else row["_depth_value"],
    ),
)

best_tree_depth = best_tree_row["_depth_value"]

tree_depth_output = tree_depth_df[
    [
        "max_depth",
        "train_rmse",
        "val_rmse",
    ]
].copy()

tree_depth_output[
    ["train_rmse", "val_rmse"]
] = tree_depth_output[
    ["train_rmse", "val_rmse"]
].round(4)

tree_depth_output.to_csv(
    "outputs/part_b_tree_depth.csv",
    index=False,
)


# --------------------------------------------------
# 6. Four candidate models
# --------------------------------------------------
models = {
    "linear_regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    ),

    "decision_tree": DecisionTreeRegressor(
        max_depth=best_tree_depth,
        random_state=42,
    ),

    "random_forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    ),

    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
    ),
}


fitted_models = {}
validation_rmse = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    fitted_models[name] = model

    val_pred = model.predict(X_val)

    metrics = regression_metrics(
        y_val,
        val_pred,
    )

    validation_rmse[name] = metrics["RMSE"]

    metric_rows.append(
        {
            "model": name,
            "split": "val",
            **metrics,
        }
    )


# --------------------------------------------------
# 7. Gradient-boosting learning-rate experiment
# --------------------------------------------------
learning_rate_rows = []

for lr in [0.01, 0.1, 1.0]:
    gb = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=lr,
        random_state=42,
    )

    gb.fit(X_train, y_train)

    val_pred = gb.predict(X_val)

    learning_rate_rows.append(
        {
            "learning_rate": lr,
            "val_rmse": np.sqrt(
                mean_squared_error(
                    y_val,
                    val_pred,
                )
            ),
            "val_r2": r2_score(
                y_val,
                val_pred,
            ),
        }
    )


learning_rate_df = pd.DataFrame(
    learning_rate_rows
)

learning_rate_df = learning_rate_df.round(4)

learning_rate_df.to_csv(
    "outputs/part_b_gb_learning_rate.csv",
    index=False,
)


# --------------------------------------------------
# 8. Cross-validation of two best validation models
# --------------------------------------------------
ranked_models = sorted(
    validation_rmse,
    key=validation_rmse.get,
)

top_two = ranked_models[:2]

X_train_val = pd.concat(
    [X_train, X_val],
    axis=0,
)

y_train_val = pd.concat(
    [y_train, y_val],
    axis=0,
)

shared_kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

cv_rows = []

for name in top_two:
    scores = cross_val_score(
        models[name],
        X_train_val,
        y_train_val,
        cv=shared_kfold,
    )

    cv_rows.append(
        {
            "model": name,
            "mean_r2": scores.mean(),
            "std_r2": scores.std(),
        }
    )


cv_df = pd.DataFrame(cv_rows).round(4)

cv_df.to_csv(
    "outputs/part_b_cv.csv",
    index=False,
)


# --------------------------------------------------
# 9. Final model — one test evaluation only
# --------------------------------------------------
final_model_name = ranked_models[0]

final_model = fitted_models[
    final_model_name
]

test_pred = final_model.predict(
    X_test
)

test_metrics = regression_metrics(
    y_test,
    test_pred,
)

metric_rows.append(
    {
        "model": final_model_name,
        "split": "test",
        **test_metrics,
    }
)


# --------------------------------------------------
# 10. Write metrics file
# --------------------------------------------------
metrics_df = pd.DataFrame(
    metric_rows,
    columns=[
        "model",
        "split",
        "MAE",
        "MSE",
        "RMSE",
        "R2",
    ],
)

for column in [
    "MAE",
    "MSE",
    "RMSE",
    "R2",
]:
    metrics_df[column] = (
        metrics_df[column].round(4)
    )

metrics_df.to_csv(
    "outputs/part_b_metrics.csv",
    index=False,
)


# --------------------------------------------------
# 11. Console summary
# --------------------------------------------------
print("\nPart B complete")

print(
    "Best tree depth:",
    "unlimited"
    if best_tree_depth is None
    else best_tree_depth,
)

print(
    "Final model:",
    final_model_name,
)

print(
    "Final test R2:",
    round(test_metrics["R2"], 4),
)

print("\nFiles written:")
print("outputs/part_b_checks.csv")
print("outputs/part_b_tree_depth.csv")
print("outputs/part_b_metrics.csv")
print("outputs/part_b_gb_learning_rate.csv")
print("outputs/part_b_cv.csv")
