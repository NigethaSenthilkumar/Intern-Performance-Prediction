import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


df = pd.read_csv("dataset/intern_performance_dataset.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nDataset Shape:", df.shape)

print("\nColumns:")
print(df.columns)


print("\nMissing Values:")
print(df.isnull().sum())

#data cleaning

df.drop_duplicates(inplace=True)
print("\nDuplicates removed")

#encoding 

df["Deadline_Met"] = df["Deadline_Met"].map({"Yes": 1, "No": 0})

df["Performance_Label"] = df["Performance_Label"].map({
    "Poor": 0,
    "Average": 1,
    "Good": 2
})

print("\nData after Encoding:")
print(df.head())


#EDA
sns.countplot(x="Performance_Label", data=df)
plt.title("Intern Performance Distribution")
plt.show()

#feature correlation
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

#save cleaned dataset

os.makedirs("output", exist_ok=True)

df.to_csv("output/cleaned_intern_dataset.csv", index=False)

print("\nCleaned dataset saved successfully!")


if "Intern_ID" in df.columns:
    X = df.drop(["Performance_Label", "Intern_ID"], axis=1)
else:
    X = df.drop("Performance_Label", axis=1)

y = df["Performance_Label"]


print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

# Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#model1- logistic regression
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)


#model2- random forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

# evaluation

from sklearn.metrics import accuracy_score, classification_report

lr_acc = accuracy_score(y_test, lr_pred)
rf_acc = accuracy_score(y_test, rf_pred)

print("\nLogistic Regression Accuracy:", lr_acc)
print("Random Forest Accuracy:", rf_acc)

print("\nRandom Forest Classification Report:\n")
print(classification_report(y_test, rf_pred))

# feature importance

print("\nFeature Importance:")

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print(feature_importance)

#confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, rf_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#gris search for hyperparameter tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nBest Parameters:", grid.best_params_)

best_pred = best_model.predict(X_test)

print("\nOptimized Accuracy:", accuracy_score(y_test, best_pred))

#model comparision
if rf_acc > lr_acc:
    print("\n✅ Random Forest performs better")
else:
    print("\n✅ Logistic Regression performs better")

#save model

import joblib

joblib.dump(rf_model, "output/model.pkl")
print("\nModel saved successfully!")

#sample

print("\nSample Predictions:", rf_pred[:5])
print("Model Score:", rf_model.score(X_test, y_test))
print("Feature Importance Values:", rf_model.feature_importances_)
print("Number of Trees:", len(rf_model.estimators_))