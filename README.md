# Intern Performance Prediction System

## 📌 Overview

This project focuses on predicting the performance of interns using Machine Learning techniques. It involves data preprocessing, exploratory data analysis (EDA), model building, evaluation, and optimization to achieve accurate predictions.

The system analyzes various features related to intern activities and predicts performance levels categorized as:

* Poor
* Average
* Good

---

## 🎯 Objectives

* Clean and preprocess real-world dataset
* Perform exploratory data analysis (EDA)
* Build and compare machine learning models
* Optimize model performance using hyperparameter tuning
* Evaluate models using standard metrics
* Save the trained model for future use

---

## 🗂️ Dataset

The dataset contains information about intern activities such as:

* Task completion
* Deadlines met
* Performance indicators

### Target Variable:

* `Performance_Label` (Encoded as 0, 1, 2)

---

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib

---

## 🔄 Workflow

### 1. Data Loading

* Dataset loaded using Pandas

### 2. Data Preprocessing

* Handling missing values
* Removing duplicate records
* Encoding categorical variables

### 3. Exploratory Data Analysis (EDA)

* Distribution of performance levels
* Feature correlation using heatmap

### 4. Feature Engineering

* Splitting features (`X`) and target (`y`)
* Removing unnecessary columns like `Intern_ID`

### 5. Data Splitting

* Train-test split (80% training, 20% testing)

### 6. Feature Scaling

* Standardization using `StandardScaler`

---

## 🤖 Models Implemented

### 1. Logistic Regression

* Used as baseline model
* Suitable for classification tasks

### 2. Random Forest Classifier

* Ensemble learning method
* Handles non-linearity effectively

---

## 📊 Model Evaluation

* Accuracy Score
* Classification Report
* Confusion Matrix

---

## ⚡ Hyperparameter Tuning

* Applied GridSearchCV on Random Forest
* Optimized parameters:

  * Number of estimators
  * Maximum depth

---

## 📈 Results

* Compared Logistic Regression and Random Forest
* Selected best-performing model based on accuracy
* Achieved improved performance after tuning

---

## 💾 Model Saving

* Final model saved using `joblib`
* File location:

```
output/model.pkl
```

---

## 📁 Project Structure

```
Intern-Performance-Prediction/
│
├── dataset/
│   └── intern_performance_dataset.csv
│
├── output/
│   ├── cleaned_intern_dataset.csv
│   └── model.pkl
│
├── main.py
└── README.md
```

---

## 🚀 How to Run

1. Install dependencies:

```
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

2. Run the script:

```
python main.py
```

---

## 🔍 Key Features

* End-to-end ML pipeline
* Data visualization for insights
* Model comparison and selection
* Hyperparameter optimization
* Model persistence for deployment

---

## 📌 Future Enhancements

* Deploy model as a web application
* Add more advanced models (XGBoost, Neural Networks)
* Improve dataset size and feature engineering
* Real-time prediction system

---

## 👩‍💻 Author

**Nigetha Senthikumar**

---

## 📜 License

This project is for academic and educational purposes.
