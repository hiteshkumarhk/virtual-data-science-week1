# ============================================
# WEEK 2 - TITANIC EDA & VISUALIZATION
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("======================================")
print("       WEEK 2 EDA STARTED")
print("======================================")

# --------------------------------------------
# 1. Load Week 1 cleaned dataset
# --------------------------------------------

df = pd.read_csv("/Users/hiteshkumar/Documents/titanic_cleaned_week1.csv")

print("\nDataset loaded successfully!")
print("Dataset Shape:", df.shape)

# --------------------------------------------
# 2. First 5 rows
# --------------------------------------------

print("\nFirst 5 Rows:")
print(df.head())

# --------------------------------------------
# 3. Basic information
# --------------------------------------------

print("\nDataset Information:")
df.info()

# --------------------------------------------
# 4. Descriptive Statistics
# --------------------------------------------

print("\nDescriptive Statistics:")
print(df.describe(include="all").T)

# --------------------------------------------
# 5. Survival Count
# --------------------------------------------

print("\nSurvival Count:")
print(df["survived"].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="survived")
plt.title("Titanic Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.show()

# --------------------------------------------
# 6. Survival Rate by Gender
# --------------------------------------------

print("\nSurvival Rate by Gender:")
print(df.groupby("sex")["survived"].mean())

plt.figure(figsize=(6, 4))
sns.barplot(data=df, x="sex", y="survived")
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")
plt.show()

# --------------------------------------------
# 7. Survival Rate by Passenger Class
# --------------------------------------------

print("\nSurvival Rate by Passenger Class:")
print(df.groupby("pclass")["survived"].mean())

plt.figure(figsize=(6, 4))
sns.barplot(data=df, x="pclass", y="survived")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.show()

# --------------------------------------------
# 8. Age Distribution
# --------------------------------------------

plt.figure(figsize=(7, 4))
sns.histplot(data=df, x="age", bins=30, kde=True)
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.show()

# --------------------------------------------
# 9. Fare Distribution
# --------------------------------------------

plt.figure(figsize=(7, 4))
sns.histplot(data=df, x="fare", bins=30, kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.show()

# --------------------------------------------
# 10. Fare by Passenger Class
# --------------------------------------------

plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="pclass", y="fare")
plt.title("Fare Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.show()

# --------------------------------------------
# 11. Age by Survival
# --------------------------------------------

plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="survived", y="age")
plt.title("Age Distribution by Survival")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Age")
plt.show()

# --------------------------------------------
# 12. Survival by Gender and Class
# --------------------------------------------

plt.figure(figsize=(7, 4))
sns.barplot(data=df, x="pclass", y="survived", hue="sex")
plt.title("Survival Rate by Class and Gender")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.legend(title="Gender")
plt.show()

# --------------------------------------------
# 13. Correlation Heatmap
# --------------------------------------------

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10, 7))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# --------------------------------------------
# 14. Fare vs Age
# --------------------------------------------

plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df,
    x="age",
    y="fare",
    hue="survived",
    alpha=0.7
)
plt.title("Age vs Fare by Survival")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.legend(title="Survived")
plt.show()

# --------------------------------------------
# 15. Embarkation Port Analysis
# --------------------------------------------

print("\nSurvival Rate by Embarkation Port:")
print(df.groupby("embarked")["survived"].mean())

plt.figure(figsize=(6, 4))
sns.barplot(data=df, x="embarked", y="survived")
plt.title("Survival Rate by Embarkation Port")
plt.xlabel("Embarkation Port")
plt.ylabel("Survival Rate")
plt.show()

# --------------------------------------------
# 16. Passenger Class Count
# --------------------------------------------

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="pclass")
plt.title("Passenger Count by Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.show()

# --------------------------------------------
# 17. Important Findings
# --------------------------------------------

print("\n======================================")
print("          KEY FINDINGS")
print("======================================")

print("\n1. Survival rate by gender:")
print(df.groupby("sex")["survived"].mean())

print("\n2. Survival rate by passenger class:")
print(df.groupby("pclass")["survived"].mean())

print("\n3. Survival rate by embarkation port:")
print(df.groupby("embarked")["survived"].mean())

print("\n4. Average fare by passenger class:")
print(df.groupby("pclass")["fare"].mean())

print("\n5. Average age by survival:")
print(df.groupby("survived")["age"].mean())

print("\n======================================")
print("       WEEK 2 EDA COMPLETED")
print("======================================")
