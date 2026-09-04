import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"

# 1. Data acquisition
df = pd.read_csv(URL)
print("Shape:", df.shape)
print(df.head())
print(df.info())

# 2. Initial exploration
print("\nDescriptive statistics:")
print(df.describe(include="all").T)

print("\nMissing values:")
missing = df.isna().sum().sort_values(ascending=False)
print(missing)

print("\nMissing percentages:")
print((missing / len(df) * 100).round(2))

print("\nDuplicate rows:", df.duplicated().sum())

# 3. Basic validation / consistency checks
print("\nUnique values for categorical columns:")
for col in ["sex", "embarked", "class", "who", "deck", "embark_town", "alive"]:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False))

# Check logical ranges
print("\nInvalid age values:", ((df["age"] < 0) | (df["age"] > 100)).sum())
print("Invalid fare values:", (df["fare"] < 0).sum())
print("Invalid pclass values:", (~df["pclass"].isin([1, 2, 3])).sum())
print("Invalid survived values:", (~df["survived"].isin([0, 1])).sum())

# 4. Missing-value treatment
# Deck is missing for about 77% of records, so it is dropped rather than
# heavily imputed. Age uses the median because it is robust to skew/outliers.
# Embarked has only two missing values, so mode imputation is appropriate.
clean = df.copy()

clean = clean.drop(columns=["deck"])

age_median = clean["age"].median()
clean["age"] = clean["age"].fillna(age_median)

embarked_mode = clean["embarked"].mode()[0]
clean["embarked"] = clean["embarked"].fillna(embarked_mode)

# embark_town corresponds directly to embarked in this dataset.
town_map = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
clean["embark_town"] = clean["embark_town"].fillna(clean["embarked"].map(town_map))

# 5. Duplicate removal
clean = clean.drop_duplicates().reset_index(drop=True)

# 6. Outlier detection using the IQR rule
def iqr_bounds(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

for col in ["age", "fare"]:
    lower, upper = iqr_bounds(clean[col])
    mask = (clean[col] < lower) | (clean[col] > upper)
    print(f"\n{col}: lower={lower:.4f}, upper={upper:.4f}, outliers={mask.sum()}")

# 7. Outlier treatment
# The extreme values are not automatically errors. For Fare, high values
# can represent genuine first-class/group tickets. We therefore retain the
# raw values in the cleaned analytical dataset and flag them instead of
# deleting them.
fare_lower, fare_upper = iqr_bounds(clean["fare"])
age_lower, age_upper = iqr_bounds(clean["age"])

clean["fare_outlier_flag"] = (
    (clean["fare"] < fare_lower) | (clean["fare"] > fare_upper)
)
clean["age_outlier_flag"] = (
    (clean["age"] < age_lower) | (clean["age"] > age_upper)
)

# 8. Optional modelling representation
# Convert selected categorical variables to category dtype.
for col in ["sex", "embarked", "class", "who", "embark_town"]:
    clean[col] = clean[col].astype("category")

# 9. Final quality checks
print("\nFinal shape:", clean.shape)
print("\nRemaining missing values:")
print(clean.isna().sum().sort_values(ascending=False))

print("\nFinal dtypes:")
print(clean.dtypes)

# 10. Save cleaned data
clean.to_csv("titanic_cleaned_week1.csv", index=False)

# 11. Optional visual checks
sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.boxplot(x=clean["fare"])
plt.title("Fare Distribution After Cleaning (Outliers Flagged, Not Deleted)")
plt.xlabel("Fare")
plt.tight_layout()
plt.savefig("fare_boxplot.png", dpi=200)
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(clean["age"], bins=30, kde=True)
plt.title("Age Distribution After Median Imputation")
plt.xlabel("Age")
plt.tight_layout()
plt.savefig("age_distribution.png", dpi=200)
plt.show()