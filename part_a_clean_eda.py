import os
import numpy as np
import pandas as pd

from src.shipment_prep import (
    FEATURES,
    clean_shipments,
    build_features,
)

os.makedirs("outputs", exist_ok=True)

# --------------------------------------------------
# 1. Read source data
# --------------------------------------------------
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# --------------------------------------------------
# 2. Metrics measured before cleaning/filling
# --------------------------------------------------
raw_rows = len(raw)

dedup = raw.drop_duplicates().copy()
rows_after_dedup = len(dedup)
duplicate_rows_removed = raw_rows - rows_after_dedup

missing_distance_km_before = int(dedup["distance_km"].isna().sum())
missing_parcel_weight_before = int(dedup["parcel_weight_kg"].isna().sum())
missing_declared_value_before = int(dedup["declared_value"].isna().sum())
missing_payment_mode_before = int(dedup["payment_mode"].isna().sum())

sentinel_weight_rows = int((dedup["parcel_weight_kg"] > 100).sum())

service_type_spellings_before = int(
    dedup["service_type"].nunique(dropna=True)
)

# --------------------------------------------------
# 3. Clean and build features
# --------------------------------------------------
clean, medians = clean_shipments(raw)

service_type_spellings_after = int(
    clean["service_type"].nunique(dropna=True)
)

modelling = build_features(clean, hubs)

rows_after_hub_merge = len(modelling)

# --------------------------------------------------
# 4. Add delayed target
# --------------------------------------------------
modelling["is_delayed"] = (
    modelling["delivery_hours"] > modelling["promised_hours"]
).astype(int)

# --------------------------------------------------
# 5. Validate required feature/target nulls
# --------------------------------------------------
required_cols = FEATURES + ["delivery_hours", "is_delayed"]

nulls_in_features_and_target = int(
    modelling[required_cols].isnull().sum().sum()
)

# --------------------------------------------------
# 6. Delivery-hours IQR checks
# --------------------------------------------------
hours_q1 = modelling["delivery_hours"].quantile(0.25)
hours_q3 = modelling["delivery_hours"].quantile(0.75)

hours_iqr = hours_q3 - hours_q1
hours_lower_bound = hours_q1 - 1.5 * hours_iqr
hours_upper_bound = hours_q3 + 1.5 * hours_iqr

hours_outlier_count = int(
    (
        (modelling["delivery_hours"] < hours_lower_bound)
        | (modelling["delivery_hours"] > hours_upper_bound)
    ).sum()
)

# --------------------------------------------------
# 7. Declared-value summary
# --------------------------------------------------
declared_value_mean_inr = modelling["declared_value_inr"].mean()
declared_value_median_inr = modelling["declared_value_inr"].median()

log_declared_value_mean = modelling["log_declared_value"].mean()
log_declared_value_median = modelling["log_declared_value"].median()

# --------------------------------------------------
# 8. Delay summary
# --------------------------------------------------
delayed_count = int(modelling["is_delayed"].sum())
delay_rate = modelling["is_delayed"].mean()

# --------------------------------------------------
# 9. Write modelling table
# --------------------------------------------------
modelling.to_csv(
    "data/modelling_table.csv",
    index=False
)

# --------------------------------------------------
# 10. Part A checks
# --------------------------------------------------
checks = [
    ("raw_rows", raw_rows),
    ("duplicate_rows_removed", duplicate_rows_removed),
    ("rows_after_dedup", rows_after_dedup),
    ("missing_distance_km_before", missing_distance_km_before),
    ("missing_parcel_weight_before", missing_parcel_weight_before),
    ("sentinel_weight_rows", sentinel_weight_rows),
    ("missing_declared_value_before", missing_declared_value_before),
    ("missing_payment_mode_before", missing_payment_mode_before),
    ("service_type_spellings_before", service_type_spellings_before),
    ("service_type_spellings_after", service_type_spellings_after),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("rows_after_hub_merge", rows_after_hub_merge),
    ("nulls_in_features_and_target", nulls_in_features_and_target),
    ("delayed_count", delayed_count),
    ("delay_rate", delay_rate),
    ("hours_q1", hours_q1),
    ("hours_q3", hours_q3),
    ("hours_lower_bound", hours_lower_bound),
    ("hours_upper_bound", hours_upper_bound),
    ("hours_outlier_count", hours_outlier_count),
    ("declared_value_mean_inr", declared_value_mean_inr),
    ("declared_value_median_inr", declared_value_median_inr),
    ("log_declared_value_mean", log_declared_value_mean),
    ("log_declared_value_median", log_declared_value_median),
]

checks_df = pd.DataFrame(checks, columns=["metric", "value"])

checks_df["value"] = checks_df["value"].apply(
    lambda x: round(float(x), 4)
    if isinstance(x, (float, np.floating))
    else x
)

checks_df.to_csv(
    "outputs/part_a_checks.csv",
    index=False
)

# --------------------------------------------------
# 11. Delay evidence by group
# --------------------------------------------------
modelling["distance_band"] = pd.cut(
    modelling["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True,
    include_lowest=True,
)

delay_tables = []

for dimension in ["service_type", "hub_tier", "distance_band"]:
    grouped = (
        modelling
        .groupby(dimension, observed=False)
        .agg(
            shipments=("shipment_id", "count"),
            delayed=("is_delayed", "sum"),
            delay_rate=("is_delayed", "mean"),
            mean_hours=("delivery_hours", "mean"),
        )
        .reset_index()
    )

    grouped.insert(0, "dimension", dimension)
    grouped = grouped.rename(columns={dimension: "group"})

    delay_tables.append(grouped)

delay_by_group = pd.concat(
    delay_tables,
    ignore_index=True
)

delay_by_group["delay_rate"] = delay_by_group["delay_rate"].round(4)
delay_by_group["mean_hours"] = delay_by_group["mean_hours"].round(4)

delay_by_group.to_csv(
    "outputs/part_a_delay_by_group.csv",
    index=False
)

# --------------------------------------------------
# 12. Correlation matrix
# --------------------------------------------------
numeric = modelling.select_dtypes(include=np.number)

corr = numeric.corr().round(4)

corr_out = corr.reset_index().rename(
    columns={"index": "variable"}
)

corr_out.to_csv(
    "outputs/part_a_correlations.csv",
    index=False
)

# --------------------------------------------------
# 13. Console confirmation
# --------------------------------------------------
print("Part A complete")
print("Rows in modelling table:", len(modelling))
print("Delayed parcels:", delayed_count)
print("Delay rate:", round(delay_rate, 4))

print("\nMedians:")
print(medians)

print("\nFiles written:")
print("data/modelling_table.csv")
print("outputs/part_a_checks.csv")
print("outputs/part_a_delay_by_group.csv")
print("outputs/part_a_correlations.csv")
