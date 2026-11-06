import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.shipment_prep import FEATURES, NOISE_COLUMNS


os.makedirs("outputs", exist_ok=True)


# --------------------------------------------------
# 1. Read modelling table
# --------------------------------------------------
df = pd.read_csv("data/modelling_table.csv")

X = df[FEATURES]
y = df["is_delayed"]


# --------------------------------------------------
# 2. Stratified train / validation / test split
# --------------------------------------------------
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

print("Train rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))

print("Delayed in train:", int(y_train.sum()))
print("Delayed in validation:", int(y_val.sum()))
print("Delayed in test:", int(y_test.sum()))


# --------------------------------------------------
# Helper: classification metrics
# --------------------------------------------------
def classification_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }


# --------------------------------------------------
# 3. Blind baseline
# --------------------------------------------------
model_rows = []

blind_pred = np.zeros(len(y_val), dtype=int)

blind_metrics = classification_metrics(
    y_val,
    blind_pred,
)

model_rows.append(
    {
        "model": "blind_all_on_time",
        **blind_metrics,
    }
)


# --------------------------------------------------
# 4. Logistic regression
# --------------------------------------------------
logistic = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression()),
    ]
)

logistic.fit(X_train, y_train)

logistic_pred = logistic.predict(X_val)

logistic_metrics = classification_metrics(
    y_val,
    logistic_pred,
)

model_rows.append(
    {
        "model": "logistic_regression",
        **logistic_metrics,
    }
)


# --------------------------------------------------
# 5. Decision-tree depth selection
# --------------------------------------------------
tree_depths = [2, 3, 4, 5, 6, 8, 10]

tree_results = []

for depth in tree_depths:
    tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42,
    )

    tree.fit(X_train, y_train)

    val_pred = tree.predict(X_val)

    val_f1 = f1_score(
        y_val,
        val_pred,
        zero_division=0,
    )

    tree_results.append(
        {
            "depth": depth,
            "f1": val_f1,
            "model": tree,
            "pred": val_pred,
        }
    )

best_tree_result = sorted(
    tree_results,
    key=lambda row: (
        -row["f1"],
        row["depth"],
    ),
)[0]

best_tree_depth = best_tree_result["depth"]

best_tree_metrics = classification_metrics(
    y_val,
    best_tree_result["pred"],
)

model_rows.append(
    {
        "model": f"decision_tree_depth_{best_tree_depth}",
        **best_tree_metrics,
    }
)


# --------------------------------------------------
# 6. Default random forest
# --------------------------------------------------
default_forest_all = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

default_forest_all.fit(
    X_train,
    y_train,
)

forest_pred = default_forest_all.predict(
    X_val
)

forest_metrics = classification_metrics(
    y_val,
    forest_pred,
)

model_rows.append(
    {
        "model": "random_forest_default",
        **forest_metrics,
    }
)


# --------------------------------------------------
# 7. Write model-comparison output
# --------------------------------------------------
models_df = pd.DataFrame(
    model_rows,
    columns=[
        "model",
        "TN",
        "FP",
        "FN",
        "TP",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ],
)

for column in [
    "accuracy",
    "precision",
    "recall",
    "f1",
]:
    models_df[column] = models_df[column].round(4)

models_df.to_csv(
    "outputs/part_c_models.csv",
    index=False,
)


# --------------------------------------------------
# 8. Feature scores: forest importance + F-test
# --------------------------------------------------
selector_all = SelectKBest(
    score_func=f_classif,
    k="all",
)

selector_all.fit(
    X_train,
    y_train,
)

f_scores = selector_all.scores_

rf_importance = default_forest_all.feature_importances_

feature_scores = pd.DataFrame(
    {
        "feature": FEATURES,
        "f_score": f_scores,
        "rf_importance": rf_importance,
    }
)

feature_scores["f_rank"] = (
    feature_scores["f_score"]
    .rank(
        method="min",
        ascending=False,
    )
    .astype(int)
)

feature_scores["is_noise"] = (
    feature_scores["feature"]
    .isin(NOISE_COLUMNS)
)

feature_scores = feature_scores[
    [
        "feature",
        "f_score",
        "f_rank",
        "rf_importance",
        "is_noise",
    ]
]

feature_scores["f_score"] = (
    feature_scores["f_score"].round(4)
)

feature_scores["rf_importance"] = (
    feature_scores["rf_importance"].round(4)
)

feature_scores.to_csv(
    "outputs/part_c_feature_scores.csv",
    index=False,
)


# --------------------------------------------------
# 9. SelectKBest experiment
# --------------------------------------------------
selectk_rows = []

for k in [5, 7, 10, 12, 17]:
    model = Pipeline(
        [
            (
                "selector",
                SelectKBest(
                    score_func=f_classif,
                    k=k,
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train,
    )

    pred = model.predict(X_val)

    selectk_rows.append(
        {
            "k": k,
            "val_f1": f1_score(
                y_val,
                pred,
                zero_division=0,
            ),
        }
    )

selectk_df = pd.DataFrame(
    selectk_rows
)

selectk_df["val_f1"] = (
    selectk_df["val_f1"].round(4)
)

selectk_df.to_csv(
    "outputs/part_c_selectk.csv",
    index=False,
)


# --------------------------------------------------
# 10. Fit selector k=7 on training rows only
# --------------------------------------------------
selector_7 = SelectKBest(
    score_func=f_classif,
    k=7,
)

X_train_7 = selector_7.fit_transform(
    X_train,
    y_train,
)

X_val_7 = selector_7.transform(
    X_val
)


# --------------------------------------------------
# 11. Grid and randomized tuning
# --------------------------------------------------
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10],
}

grid_search = GridSearchCV(
    RandomForestClassifier(
        random_state=42
    ),
    param_grid=param_grid,
    cv=5,
)

grid_search.fit(
    X_train_7,
    y_train,
)

grid_val_pred = grid_search.best_estimator_.predict(
    X_val_7
)

grid_val_f1 = f1_score(
    y_val,
    grid_val_pred,
    zero_division=0,
)


randomized_search = RandomizedSearchCV(
    RandomForestClassifier(
        random_state=42
    ),
    param_distributions=param_grid,
    n_iter=9,
    cv=5,
    random_state=42,
)

randomized_search.fit(
    X_train_7,
    y_train,
)

random_val_pred = (
    randomized_search.best_estimator_.predict(
        X_val_7
    )
)

random_val_f1 = f1_score(
    y_val,
    random_val_pred,
    zero_division=0,
)


default_forest_7 = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

default_forest_7.fit(
    X_train_7,
    y_train,
)

default_val_pred = default_forest_7.predict(
    X_val_7
)

default_val_f1 = f1_score(
    y_val,
    default_val_pred,
    zero_division=0,
)


# --------------------------------------------------
# 12. Tuning output
# --------------------------------------------------
tuning_rows = [
    {
        "search": "grid_search",
        "candidates": 27,
        "cv_folds": 5,
        "total_fits": 135,
        "best_params": json.dumps(
            grid_search.best_params_,
            sort_keys=True,
        ),
        "best_cv_accuracy": grid_search.best_score_,
        "val_f1": grid_val_f1,
    },
    {
        "search": "randomized_search",
        "candidates": 9,
        "cv_folds": 5,
        "total_fits": 45,
        "best_params": json.dumps(
            randomized_search.best_params_,
            sort_keys=True,
        ),
        "best_cv_accuracy": randomized_search.best_score_,
        "val_f1": random_val_f1,
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": "defaults",
        "best_cv_accuracy": np.nan,
        "val_f1": default_val_f1,
    },
]

tuning_df = pd.DataFrame(
    tuning_rows
)

tuning_df["best_cv_accuracy"] = (
    tuning_df["best_cv_accuracy"].round(4)
)

tuning_df["val_f1"] = (
    tuning_df["val_f1"].round(4)
)

tuning_df.to_csv(
    "outputs/part_c_tuning.csv",
    index=False,
)


# --------------------------------------------------
# 13. Choose final configuration
# Highest validation F1.
# Ties: default, then grid, then randomized.
# --------------------------------------------------
priority = {
    "default_forest_same_7_features": 0,
    "grid_search": 1,
    "randomized_search": 2,
}

choice_rows = [
    {
        "name": "grid_search",
        "val_f1": grid_val_f1,
        "params": grid_search.best_params_,
    },
    {
        "name": "randomized_search",
        "val_f1": random_val_f1,
        "params": randomized_search.best_params_,
    },
    {
        "name": "default_forest_same_7_features",
        "val_f1": default_val_f1,
        "params": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_leaf": 1,
        },
    },
]

chosen = sorted(
    choice_rows,
    key=lambda row: (
        -row["val_f1"],
        priority[row["name"]],
    ),
)[0]

chosen_from = chosen["name"]
chosen_params = chosen["params"]


# --------------------------------------------------
# 14. Final pipeline, fitted on TRAINING only
# --------------------------------------------------
final_model = Pipeline(
    [
        (
            "selector",
            SelectKBest(
                score_func=f_classif,
                k=7,
            ),
        ),
        (
            "model",
            RandomForestClassifier(
                random_state=42,
                n_estimators=chosen_params[
                    "n_estimators"
                ],
                max_depth=chosen_params[
                    "max_depth"
                ],
                min_samples_leaf=chosen_params[
                    "min_samples_leaf"
                ],
            ),
        ),
    ]
)

final_model.fit(
    X_train,
    y_train,
)


# --------------------------------------------------
# 15. Validation probabilities and threshold table
# --------------------------------------------------
val_probability = final_model.predict_proba(
    X_val
)[:, 1]

thresholds = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
]

threshold_rows = []

for threshold in thresholds:
    pred = (
        val_probability >= threshold
    ).astype(int)

    metrics = classification_metrics(
        y_val,
        pred,
    )

    threshold_rows.append(
        {
            "threshold": threshold,
            **metrics,
        }
    )

threshold_df = pd.DataFrame(
    threshold_rows,
    columns=[
        "threshold",
        "TN",
        "FP",
        "FN",
        "TP",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ],
)

for column in [
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
]:
    threshold_df[column] = (
        threshold_df[column].round(4)
    )

threshold_df.to_csv(
    "outputs/part_c_thresholds.csv",
    index=False,
)


# --------------------------------------------------
# 16. Choose highest threshold with recall >= 0.80
# --------------------------------------------------
eligible = threshold_df[
    threshold_df["recall"] >= 0.80
]

if len(eligible) > 0:
    chosen_threshold = float(
        eligible["threshold"].max()
    )
else:
    chosen_threshold = 0.1


# --------------------------------------------------
# 17. One test evaluation only
# --------------------------------------------------
test_probability = final_model.predict_proba(
    X_test
)[:, 1]

test_pred = (
    test_probability >= chosen_threshold
).astype(int)

test_metrics = classification_metrics(
    y_test,
    test_pred,
)

final_test_df = pd.DataFrame(
    [
        {
            "threshold": chosen_threshold,
            **test_metrics,
        }
    ]
)

for column in [
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
]:
    final_test_df[column] = (
        final_test_df[column].round(4)
    )

final_test_df.to_csv(
    "outputs/part_c_final_test.csv",
    index=False,
)


# --------------------------------------------------
# 18. Final configuration output
# --------------------------------------------------
max_depth_value = (
    "unlimited"
    if chosen_params["max_depth"] is None
    else chosen_params["max_depth"]
)

final_config_df = pd.DataFrame(
    [
        {
            "selector_k": 7,
            "n_estimators": chosen_params[
                "n_estimators"
            ],
            "max_depth": max_depth_value,
            "min_samples_leaf": chosen_params[
                "min_samples_leaf"
            ],
            "threshold": chosen_threshold,
            "chosen_from": chosen_from,
        }
    ]
)

final_config_df.to_csv(
    "outputs/part_c_final_config.csv",
    index=False,
)


# --------------------------------------------------
# 19. Console summary
# --------------------------------------------------
print("\nPart C complete")

print(
    "Best decision tree depth:",
    best_tree_depth,
)

print(
    "Chosen tuning configuration:",
    chosen_from,
)

print(
    "Chosen threshold:",
    chosen_threshold,
)

print(
    "Test precision:",
    round(
        test_metrics["precision"],
        4,
    ),
)

print(
    "Test recall:",
    round(
        test_metrics["recall"],
        4,
    ),
)

print(
    "Test F1:",
    round(
        test_metrics["f1"],
        4,
    ),
)

print("\nFiles written:")
print("outputs/part_c_models.csv")
print("outputs/part_c_feature_scores.csv")
print("outputs/part_c_selectk.csv")
print("outputs/part_c_tuning.csv")
print("outputs/part_c_thresholds.csv")
print("outputs/part_c_final_test.csv")
print("outputs/part_c_final_config.csv")
