import os
import itertools
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


os.makedirs("outputs", exist_ok=True)


# --------------------------------------------------
# 1. Read modelling table
# --------------------------------------------------
df = pd.read_csv("data/modelling_table.csv")


# --------------------------------------------------
# 2. Build hub profile
# --------------------------------------------------
hub_profile = (
    df.groupby("hub_id")
    .agg(
        shipments=("shipment_id", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean"),
    )
    .reset_index()
    .sort_values("hub_id")
    .reset_index(drop=True)
)


hub_profile_output = hub_profile.copy()

for column in [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops",
]:
    hub_profile_output[column] = (
        hub_profile_output[column].round(4)
    )

hub_profile_output.to_csv(
    "outputs/hub_profile.csv",
    index=False,
)


# --------------------------------------------------
# 3. Scale hub-level clustering variables
# --------------------------------------------------
profile_columns = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops",
]

scaler = StandardScaler()

scaled_profile = scaler.fit_transform(
    hub_profile[profile_columns]
)


# --------------------------------------------------
# Helper: across-to-within pairwise distance ratio
# --------------------------------------------------
def separation_ratio(matrix, labels):
    within_distances = []
    across_distances = []

    for i, j in itertools.combinations(
        range(len(matrix)),
        2,
    ):
        distance = np.linalg.norm(
            matrix[i] - matrix[j]
        )

        if labels[i] == labels[j]:
            within_distances.append(distance)
        else:
            across_distances.append(distance)

    mean_within = np.mean(within_distances)
    mean_across = np.mean(across_distances)

    return mean_across / mean_within


# --------------------------------------------------
# 4. Separation evidence for K = 2, 3, 4, 5
# --------------------------------------------------
separation_rows = []

for k in [2, 3, 4, 5]:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )

    labels = kmeans.fit_predict(
        scaled_profile
    )

    ratio = separation_ratio(
        scaled_profile,
        labels,
    )

    cluster_sizes = (
        pd.Series(labels)
        .value_counts()
        .sort_values(ascending=False)
        .tolist()
    )

    cluster_sizes_text = "-".join(
        str(size)
        for size in cluster_sizes
    )

    separation_rows.append(
        {
            "K": k,
            "across_over_within": ratio,
            "cluster_sizes": cluster_sizes_text,
        }
    )


separation_df = pd.DataFrame(
    separation_rows
)

separation_df[
    "across_over_within"
] = separation_df[
    "across_over_within"
].round(4)

separation_df.to_csv(
    "outputs/part_d_separation.csv",
    index=False,
)


# --------------------------------------------------
# 5. Final K = 3 segmentation
# --------------------------------------------------
kmeans_3 = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)

hub_profile["cluster"] = (
    kmeans_3.fit_predict(
        scaled_profile
    )
)


# --------------------------------------------------
# 6. Name segments by cluster delay rate
# Lowest = Standard
# Middle = Watch
# Highest = Escalate
# --------------------------------------------------
cluster_delay = (
    hub_profile
    .groupby("cluster")["delay_rate"]
    .mean()
    .sort_values()
)

ordered_clusters = (
    cluster_delay.index.tolist()
)

segment_mapping = {
    ordered_clusters[0]: "Standard",
    ordered_clusters[1]: "Watch",
    ordered_clusters[2]: "Escalate",
}

hub_profile["segment"] = (
    hub_profile["cluster"]
    .map(segment_mapping)
)


# --------------------------------------------------
# 7. Attach hub metadata
# --------------------------------------------------
hub_metadata = (
    df[
        [
            "hub_id",
            "hub_city",
            "hub_tier",
        ]
    ]
    .drop_duplicates("hub_id")
)

hub_segments = (
    hub_profile[
        [
            "hub_id",
            "segment",
        ]
    ]
    .merge(
        hub_metadata,
        on="hub_id",
        how="left",
    )
    .sort_values("hub_id")
    .reset_index(drop=True)
)

hub_segments.to_csv(
    "outputs/hub_segments.csv",
    index=False,
)


# --------------------------------------------------
# 8. Segment profile
# Means are calculated from hub-level values
# --------------------------------------------------
segment_order = [
    "Standard",
    "Watch",
    "Escalate",
]

segment_profile_rows = []

for segment in segment_order:
    group = hub_profile[
        hub_profile["segment"] == segment
    ]

    segment_profile_rows.append(
        {
            "segment": segment,
            "hubs": len(group),
            "mean_delivery_hours": (
                group["mean_delivery_hours"].mean()
            ),
            "delay_rate": (
                group["delay_rate"].mean()
            ),
            "mean_distance_km": (
                group["mean_distance_km"].mean()
            ),
            "mean_n_stops": (
                group["mean_n_stops"].mean()
            ),
        }
    )


segment_profile = pd.DataFrame(
    segment_profile_rows
)

for column in [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops",
]:
    segment_profile[column] = (
        segment_profile[column].round(4)
    )

segment_profile.to_csv(
    "outputs/segment_profile.csv",
    index=False,
)


# --------------------------------------------------
# 9. Compare existing tier with operating segment
# --------------------------------------------------
tier_expected_segment = {
    1: "Standard",
    2: "Watch",
    3: "Escalate",
}

hub_segments[
    "expected_segment"
] = hub_segments[
    "hub_tier"
].map(tier_expected_segment)

mismatch = hub_segments[
    hub_segments["segment"]
    != hub_segments["expected_segment"]
][
    [
        "hub_id",
        "segment",
        "hub_city",
        "hub_tier",
    ]
].copy()

mismatch.to_csv(
    "outputs/tier_vs_segment_mismatch.csv",
    index=False,
)


# --------------------------------------------------
# 10. Console summary
# --------------------------------------------------
print("Part D hub segmentation complete")

print("\nSegment counts:")
print(
    hub_segments["segment"]
    .value_counts()
)

print("\nTier / segment mismatches:")
print(mismatch)

print("\nFiles written:")
print("outputs/hub_profile.csv")
print("outputs/part_d_separation.csv")
print("outputs/hub_segments.csv")
print("outputs/segment_profile.csv")
print("outputs/tier_vs_segment_mismatch.csv")
