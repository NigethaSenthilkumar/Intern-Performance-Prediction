import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("\nLoading Dataset...\n")

df = pd.read_csv("dataset/intern_performance_dataset.csv")

print("First 5 rows of dataset:\n")
print(df.head())


print("\nDataset Information:\n")
print(df.info())

print("\nDataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns)


print("\nMissing Values in Dataset:\n")
print(df.isnull().sum())

df.drop_duplicates(inplace=True)
print("\nDuplicates removed")

df["Deadline_Met"] = df["Deadline_Met"].map({
    "Yes": 1,
    "No": 0
})

df["Performance_Label"] = df["Performance_Label"].map({
    "Poor": 0,
    "Average": 1,
    "Good": 2
})

print("\nEncoded Dataset Preview:\n")
print(df.head())

print("\nStatistical Summary:\n")
print(df.describe())

print("\nFeature Correlation Matrix:\n")
print(df.corr())

plt.figure(figsize=(6,4))
sns.countplot(x="Performance_Label", data=df)
plt.title("Intern Performance Distribution")
plt.xlabel("Performance Level")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

df.hist(figsize=(10,8))
plt.suptitle("Feature Distribution")
plt.show()

X = df.drop("Performance_Label", axis=1)

y = df["Performance_Label"]

print("\nFeature Variables:\n")
print(X.head())

print("\nTarget Variable:\n")
print(y.head())

os.makedirs("output", exist_ok=True)

df.to_csv("output/cleaned_intern_dataset.csv", index=False)

print("\nCleaned dataset saved successfully!")
 
 
print("\nFinal Dataset Structure:\n")
print(df.info())