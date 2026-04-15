# Comparing the Performance of a Single Layer Neural Network to Linear Logistic Regression

In this project we compare the performance of a single layered neural network with dropout regularization against linear logistic regression. The classification task is to
predict wether customers will default on their credit card debt. The dataset is included in the files since it comes from the book Introduction to Statistical Learning by James, Witten, Hastie, Tibshirani and Taylor, which is publicly available.

## Project Structure 

- `src` - source for project architecture, models and evaluations;
- `reports` - detailed report of the results obtained with visual plots.

## The Dataset

The dataset is a simulated data set containing information on ten thousand customers. The response variable is `default` that takes the value `yes`, in case a customer
defaulted on their credit card debt, and takes the value `no` if a customer did not default on their credit card debt. For more information, see `reports_and_results/report.md`.

## Methods

The dataset was preprocessed by removing missing values and separating predictors from the response variable. Predictors were classified as numerical or categorical. The data was split into training and test sets using `train_test_split` from `sklearn.model_selection`.

Numerical features were standardised, and categorical features were one-hot encoded using `ColumnTransformer` from `sklearn.compose`.

---

## Results Summary

The logistic regression model achieved an accuracy of **96.95%** on the test set. However, performance on the customers who defaulted was poor due to the small recall score:

- **Recall (default = Yes): ~27.5%**;
- Correctly identified **19 out of 69** defaulters;
- Missed approximately **72.5%** of actual defaults.  

This indicates strong class imbalance, with the model biased toward predicting non-default. This makes the model not indicated for real-world credit risk assessment as it is innefective predicting high-risk customers.



## How to run 

### 1) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\Activate.ps1 # Windows PowerShell
```

### 2) Install Dependencies
```bash 
pip install -r requirements.txt
```

### 3) Run the Project 
```bash
python src/main.py
```
