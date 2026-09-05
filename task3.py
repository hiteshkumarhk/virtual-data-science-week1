# ============================================================
# WEEK 3 TASK: UNSUPERVISED LEARNING AND CLUSTERING ANALYSIS
# Project: Customer Segmentation Using K-Means Clustering
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# 1. CREATE OUTPUT FOLDER
# ============================================================

output_folder = "week3_results"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("=" * 65)
print("WEEK 3 - CUSTOMER SEGMENTATION USING K-MEANS")
print("=" * 65)


# ============================================================
# 2. LOAD DATASET
# ============================================================

url = (
    "https://raw.githubusercontent.com/"
    "sharmaroshan/Clustering-of-Mall-Customers/"
    "master/Mall_Customers.csv"
)

print("\nDownloading dataset...")

try:
    df = pd.read_csv(url)
    print("Dataset downloaded successfully!")

except Exception as e:
    print("Error while downloading dataset.")
    print("Error:", e)
    exit()


# ============================================================
# 3. BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 65)
print("DATASET INFORMATION")
print("=" * 65)

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())


# ============================================================
# 4. CHECK MISSING VALUES AND DUPLICATES
# ============================================================

print("\n" + "=" * 65)
print("DATA QUALITY CHECK")
print("=" * 65)

missing_values = df.isnull().sum()

print("\nMissing values:")
print(missing_values)

duplicate_count = df.duplicated().sum()

print("\nDuplicate rows:", duplicate_count)


# ============================================================
# 5. REMOVE DUPLICATES IF ANY
# ============================================================

if duplicate_count > 0:
    df = df.drop_duplicates()
    print("\nDuplicate rows removed.")

else:
    print("\nNo duplicate rows found.")


# ============================================================
# 6. SELECT FEATURES FOR CLUSTERING
# ============================================================

print("\n" + "=" * 65)
print("FEATURE SELECTION")
print("=" * 65)

# We use Annual Income and Spending Score.
# CustomerID is not useful for clustering because it is
# only an identifier.

X = df[
    [
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
]

print("\nSelected features:")
print(X.head())


# ============================================================
# 7. STANDARDIZE FEATURES
# ============================================================

print("\n" + "=" * 65)
print("FEATURE SCALING")
print("=" * 65)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Features standardized successfully.")


# ============================================================
# 8. VISUALIZE ORIGINAL DATA
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    X["Annual Income (k$)"],
    X["Spending Score (1-100)"]
)

plt.title("Customer Distribution Before Clustering")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "01_customer_distribution_before_clustering.png"
    ),
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# 9. ELBOW METHOD
# ============================================================

print("\n" + "=" * 65)
print("ELBOW METHOD")
print("=" * 65)

wcss = []

k_values = range(2, 11)

for k in k_values:

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    wcss.append(model.inertia_)


for k, value in zip(k_values, wcss):
    print(
        f"K = {k:2d}  |  "
        f"WCSS = {value:.2f}"
    )


# Plot Elbow Method

plt.figure(figsize=(9, 6))

plt.plot(
    list(k_values),
    wcss,
    marker="o"
)

plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.xticks(list(k_values))
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "02_elbow_method.png"
    ),
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# 10. SILHOUETTE SCORE
# ============================================================

print("\n" + "=" * 65)
print("SILHOUETTE SCORE ANALYSIS")
print("=" * 65)

silhouette_scores = []

for k in k_values:

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores.append(score)

    print(
        f"K = {k:2d}  |  "
        f"Silhouette Score = {score:.4f}"
    )


# Find best K based on silhouette score

best_k = list(k_values)[
    np.argmax(silhouette_scores)
]

best_score = max(silhouette_scores)

print("\nBest K according to Silhouette Score:")
print(best_k)

print("\nHighest Silhouette Score:")
print(round(best_score, 4))


# Plot silhouette scores

plt.figure(figsize=(9, 6))

plt.plot(
    list(k_values),
    silhouette_scores,
    marker="o"
)

plt.title("Silhouette Score for Different Values of K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_values))
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "03_silhouette_scores.png"
    ),
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# 11. FINAL K-MEANS MODEL
# ============================================================

# For this project, K = 5 is used because it provides
# meaningful customer segmentation for this dataset.

optimal_k = 5

print("\n" + "=" * 65)
print("FINAL K-MEANS MODEL")
print("=" * 65)

print("\nNumber of clusters selected:", optimal_k)

kmeans = KMeans(
    n_clusters=optimal_k,
    init="k-means++",
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(X_scaled)

df["Cluster"] = cluster_labels

print("K-Means clustering completed successfully!")


# ============================================================
# 12. FINAL CLUSTER VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 7))

for cluster in range(optimal_k):

    cluster_data = df[
        df["Cluster"] == cluster
    ]

    plt.scatter(
        cluster_data["Annual Income (k$)"],
        cluster_data["Spending Score (1-100)"],
        label=f"Cluster {cluster}"
    )


# Convert centroids back to original scale

centers_original = scaler.inverse_transform(
    kmeans.cluster_centers_
)

plt.scatter(
    centers_original[:, 0],
    centers_original[:, 1],
    marker="X",
    s=250,
    label="Centroids"
)

plt.title("Customer Segmentation Using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "04_kmeans_customer_segments.png"
    ),
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# 13. CLUSTER SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("CLUSTER SUMMARY")
print("=" * 65)

cluster_summary = df.groupby(
    "Cluster"
).agg(
    Number_of_Customers=(
        "CustomerID",
        "count"
    ),

    Average_Age=(
        "Age",
        "mean"
    ),

    Average_Income=(
        "Annual Income (k$)",
        "mean"
    ),

    Average_Spending_Score=(
        "Spending Score (1-100)",
        "mean"
    )
).round(2)

print(cluster_summary)


# Save cluster summary

cluster_summary.to_csv(
    os.path.join(
        output_folder,
        "cluster_summary.csv"
    )
)


# ============================================================
# 14. CREATE MEANINGFUL CUSTOMER SEGMENTS
# ============================================================

summary = df.groupby(
    "Cluster"
).agg(
    Avg_Income=(
        "Annual Income (k$)",
        "mean"
    ),

    Avg_Spending=(
        "Spending Score (1-100)",
        "mean"
    ),

    Customers=(
        "CustomerID",
        "count"
    )
).round(2)


income_median = summary["Avg_Income"].median()

spending_median = summary["Avg_Spending"].median()

segment_names = {}


for cluster in summary.index:

    income = summary.loc[
        cluster,
        "Avg_Income"
    ]

    spending = summary.loc[
        cluster,
        "Avg_Spending"
    ]


    if (
        income >= income_median
        and spending >= spending_median
    ):

        name = "High Income - High Spending"


    elif (
        income >= income_median
        and spending < spending_median
    ):

        name = "High Income - Low Spending"


    elif (
        income < income_median
        and spending >= spending_median
    ):

        name = "Low Income - High Spending"


    else:

        name = "Low Income - Low Spending"


    segment_names[cluster] = name


df["Customer Segment"] = df[
    "Cluster"
].map(segment_names)


# ============================================================
# 15. PRINT SEGMENT MAPPING
# ============================================================

print("\n" + "=" * 65)
print("CUSTOMER SEGMENT MAPPING")
print("=" * 65)

for cluster, name in segment_names.items():

    print(
        f"Cluster {cluster} --> {name}"
    )


# ============================================================
# 16. FINAL CUSTOMER SEGMENT SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("FINAL CUSTOMER SEGMENT SUMMARY")
print("=" * 65)

final_summary = df.groupby(
    "Customer Segment"
).agg(
    Customers=(
        "CustomerID",
        "count"
    ),

    Average_Age=(
        "Age",
        "mean"
    ),

    Average_Income=(
        "Annual Income (k$)",
        "mean"
    ),

    Average_Spending_Score=(
        "Spending Score (1-100)",
        "mean"
    )
).round(2)


print(final_summary)


# Save final summary

final_summary.to_csv(
    os.path.join(
        output_folder,
        "customer_segment_summary.csv"
    )
)


# ============================================================
# 17. CUSTOMER COUNT BY SEGMENT
# ============================================================

plt.figure(figsize=(10, 6))

df[
    "Customer Segment"
].value_counts().plot(
    kind="bar"
)

plt.title(
    "Number of Customers in Each Segment"
)

plt.xlabel("Customer Segment")

plt.ylabel("Number of Customers")

plt.xticks(
    rotation=30,
    ha="right"
)

plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "05_customer_count_by_segment.png"
    ),
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# 18. SAVE COMPLETE CLUSTERED DATASET
# ============================================================

df.to_csv(
    os.path.join(
        output_folder,
        "Mall_Customers_Clustered.csv"
    ),
    index=False
)


# ============================================================
# 19. SAVE RESULTS TO TEXT FILE
# ============================================================

results_file = os.path.join(
    output_folder,
    "results.txt"
)

with open(
    results_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "WEEK 3 - CUSTOMER SEGMENTATION "
        "USING K-MEANS\n"
    )

    file.write("=" * 65 + "\n\n")

    file.write(
        f"Dataset Shape: {df.shape}\n"
    )

    file.write(
        f"Optimal K Used: {optimal_k}\n"
    )

    file.write(
        f"Best K According to Silhouette Score: "
        f"{best_k}\n"
    )

    file.write(
        f"Highest Silhouette Score: "
        f"{best_score:.4f}\n\n"
    )

    file.write(
        "SILHOUETTE SCORES\n"
    )

    file.write("-" * 40 + "\n")

    for k, score in zip(
        k_values,
        silhouette_scores
    ):

        file.write(
            f"K = {k}: "
            f"{score:.4f}\n"
        )

    file.write("\n\n")

    file.write(
        "CLUSTER SUMMARY\n"
    )

    file.write("-" * 40 + "\n")

    file.write(
        cluster_summary.to_string()
    )

    file.write("\n\n")

    file.write(
        "CUSTOMER SEGMENT SUMMARY\n"
    )

    file.write("-" * 40 + "\n")

    file.write(
        final_summary.to_string()
    )


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 65)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 65)

print("\nAll results have been saved inside:")
print(f"  {output_folder}/")

print("\nGenerated files:")

print("  1. 01_customer_distribution_before_clustering.png")
print("  2. 02_elbow_method.png")
print("  3. 03_silhouette_scores.png")
print("  4. 04_kmeans_customer_segments.png")
print("  5. 05_customer_count_by_segment.png")
print("  6. cluster_summary.csv")
print("  7. customer_segment_summary.csv")
print("  8. Mall_Customers_Clustered.csv")
print("  9. results.txt")

print("\nFinal K used:", optimal_k)
print("Silhouette Score for K=5:",
      round(
          silhouette_score(
              X_scaled,
              df["Cluster"]
          ),
          4
      ))

print("\nWeek 3 analysis is complete! ")