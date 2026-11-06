import numpy as np
import pandas as pd


PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96
}


FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]


NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]


def clean_shipments(raw, medians=None):
    clean = raw.copy()

    # 1. Remove exact duplicate rows
    clean = clean.drop_duplicates().copy()

    # 2. Standardize service type
    clean["service_type"] = (
        clean["service_type"]
        .str.strip()
        .str.title()
    )

    # 3. Fill missing payment mode
    clean["payment_mode"] = clean["payment_mode"].fillna("Unknown")

    # 4. Convert declared value text to numeric INR
    clean["declared_value_inr"] = (
        clean["declared_value"]
        .astype("string")
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
    )

    clean["declared_value_inr"] = pd.to_numeric(
        clean["declared_value_inr"],
        errors="coerce"
    )

    clean = clean.drop(columns=["declared_value"])

    # 5. Convert dispatch time to datetime
    clean["dispatch_time"] = pd.to_datetime(clean["dispatch_time"])

    # 6. Treat parcel weights above 100 kg as scanner-fault sentinels
    clean.loc[
        clean["parcel_weight_kg"] > 100,
        "parcel_weight_kg"
    ] = np.nan

    # 7. Compute or reuse medians
    if medians is None:
        medians = {
            "distance_km": float(clean["distance_km"].median()),
            "parcel_weight_kg": float(clean["parcel_weight_kg"].median()),
            "declared_value_inr": float(clean["declared_value_inr"].median()),
        }
    else:
        medians = {
            "distance_km": float(medians["distance_km"]),
            "parcel_weight_kg": float(medians["parcel_weight_kg"]),
            "declared_value_inr": float(medians["declared_value_inr"]),
        }

    # 8. Fill missing numeric values
    for column in [
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr",
    ]:
        clean[column] = clean[column].fillna(medians[column])

    return clean, medians


def build_features(clean, hubs):
    hub_columns = [
        "hub_id",
        "hub_city",
        "region",
        "hub_tier",
        "hub_daily_capacity",
    ]

    features = clean.merge(
        hubs[hub_columns],
        on="hub_id",
        how="left"
    )

    features["promised_hours"] = (
        features["service_type"]
        .map(PROMISED_HOURS)
        .astype(int)
    )

    features["is_cod"] = (
        features["payment_mode"] == "COD"
    ).astype(int)

    features["is_weekend"] = (
        features["dispatch_time"].dt.dayofweek >= 5
    ).astype(int)

    features["dispatch_hour"] = (
        features["dispatch_time"].dt.hour.astype(int)
    )

    features["log_declared_value"] = np.log(
        features["declared_value_inr"]
    )

    return features
