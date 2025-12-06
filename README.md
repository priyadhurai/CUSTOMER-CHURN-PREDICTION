# CUSTOMER-CHURN-PREDICTION

CUSTOMER CHURN PREDICTION – MINI PROJECT (WEEK 4)
Overview

The goal is to predict whether a customer will churn (leave the company) using historical customer data such as age, tenure, monthly charges, and total charges.

This project demonstrates a complete end-to-end machine learning workflow:

Data loading

Data cleaning

Feature engineering

Train/test split

Model training

Model evaluation

Visualization

This type of project is commonly used in ML interviews for data science and machine learning engineer roles.

Project Structure
customer-churn-project/
│
├── customer_churn.csv
├── customer_churn_model.py
├── feature_importance.png     (generated after running the script)
└── README.md

Dataset Description

The dataset contains basic customer attributes:

Column	Description
CustomerID	Unique customer identifier
Gender	Male/Female
Age	Customer age
Tenure	Number of months stayed with company
MonthlyCharges	Monthly bill amount
TotalCharges	Total amount paid to date
Churn	Whether customer left (1 = churn, 0 = stay)
Technologies Used

Python 3

Pandas – data cleaning and preprocessing

Scikit-learn – building ML model

Matplotlib – visualization

RandomForestClassifier – classification model

Installation

Install required Python packages:

pip install pandas scikit-learn matplotlib

How to Run the Project

Clone or download the project folder.

Make sure the dataset file customer_churn.csv is in the same directory.

Run:

python customer_churn_model.py


The script will output:

Model accuracy

Confusion matrix

Precision, Recall, F1 Score

Classification report

A feature importance bar chart (saved as feature_importance.png)

Model Workflow
1. Load Data

Reads the CSV file and displays the first few rows.

2. Data Cleaning

Converts TotalCharges to numeric

Fills missing values with the median

3. Encoding

Encodes categorical features using Label Encoding.

4. Feature Selection

Uses the following features:

Gender, Age, Tenure, MonthlyCharges, TotalCharges

5. Train-Test Split

Splits data into:

80% Training

20% Testing

6. Model Training

Uses RandomForestClassifier because it:

Works well on tabular data

Handles non-linear relationships

Is more accurate than Linear/Logistic Regression for this dataset

7. Model Evaluation

The script prints:

Accuracy

Confusion matrix

Precision/Recall/F1 Score

8. Visualization

Generates a bar chart showing feature importance.

Sample Output (Example)
Accuracy: 0.75

Confusion Matrix:
[[2 1]
 [0 1]]

Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.67      0.80         3
           1       0.50      1.00      0.67         1


Feature importance plot is generated as:

feature_importance.png
