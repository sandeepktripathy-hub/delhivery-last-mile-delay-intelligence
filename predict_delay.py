import joblib
import pandas as pd

from src.shipment_prep import clean_shipments, build_features


# --------------------------------------------------
# 1. Load required source files
# --------------------------------------------------
new_shipments = pd.read_csv(
    "data/new_shipments.csv"
)

hubs = pd.read_csv(
    "data/hubs.csv"
)

hub_segments = pd.read_csv(
    "outputs/hub_segments.csv"
)


# --------------------------------------------------
# 2. Load packaged model artefacts
# --------------------------------------------------
model = joblib.load(
    "models/delay_model.pkl"
)

threshold = float(
    pd.read_csv(
        "models/delay_threshold.csv"
    )["threshold"].iloc[0]
)

feature_order = (
    pd.read_csv(
        "models/feature_order.csv"
    )["feature"]
    .tolist()
)

impute_df = pd.read_csv(
    "models/impute_values.csv"
)

medians = dict(
    zip(
        impute_df["column"],
        impute_df["median"],
    )
)


# --------------------------------------------------
# 3. Clean new shipments using stored medians
# --------------------------------------------------
clean_new, _ = clean_shipments(
    new_shipments,
    medians=medians,
)


# --------------------------------------------------
# 4. Build model features
# --------------------------------------------------
new_features = build_features(
    clean_new,
    hubs,
)

X_new = new_features[
    feature_order
]


# --------------------------------------------------
# 5. Predict delay probability
# --------------------------------------------------
delay_probability = (
    model.predict_proba(
        X_new
    )[:, 1]
)

flagged = (
    delay_probability >= threshold
).astype(int)


# --------------------------------------------------
# 6. Join hub operating segment
# --------------------------------------------------
scored = new_features[
    [
        "shipment_id",
        "hub_id",
    ]
].copy()

scored["delay_probability"] = (
    delay_probability
)

scored["flagged"] = flagged

scored = scored.merge(
    hub_segments[
        [
            "hub_id",
            "segment",
        ]
    ],
    on="hub_id",
    how="left",
)


# --------------------------------------------------
# 7. Operational action rule
# --------------------------------------------------
def choose_action(row):
    if row["flagged"] == 0:
        return "No action"

    if row["segment"] == "Escalate":
        return "Escalate to hub manager"

    return "Proactive customer call"


scored["action"] = scored.apply(
    choose_action,
    axis=1,
)


# --------------------------------------------------
# 8. Final required column order
# --------------------------------------------------
scored["delay_probability"] = (
    scored["delay_probability"].round(4)
)

output = scored[
    [
        "shipment_id",
        "hub_id",
        "segment",
        "delay_probability",
        "flagged",
        "action",
    ]
]


# --------------------------------------------------
# 9. Save output
# --------------------------------------------------
output.to_csv(
    "outputs/new_shipment_actions.csv",
    index=False,
)


# --------------------------------------------------
# 10. Console summary
# --------------------------------------------------
print("New shipment scoring complete")

print(
    "Rows scored:",
    len(output),
)

print(
    "Flagged shipments:",
    int(output["flagged"].sum()),
)

print(
    "Hub-manager escalations:",
    int(
        (
            output["action"]
            == "Escalate to hub manager"
        ).sum()
    ),
)

print("\nAction counts:")
print(
    output["action"]
    .value_counts()
)

print(
    "\nFile written:"
)
print(
    "outputs/new_shipment_actions.csv"
)
