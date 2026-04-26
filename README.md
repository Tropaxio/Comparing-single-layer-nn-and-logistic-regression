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

For the neural network, dataset was converted from Pandas DataFrames into PyTorch tensors. The target variable was reshaped to match the expected input of the loss function. The data was then wrapped into TensorDataset objects and loaded with DataLoader to enable efficient mini-batch training (batch size = 32) with shuffling applied.

---

## Results Summary

### Logistic Regression

The logistic regression model achieved an accuracy of **96.95%** on the test set. However, performance on the customers who defaulted was poor due to the small recall score:

- **Recall (default = Yes): ~27.5%**;
- Correctly identified **19 out of 69** defaulters;
- Missed approximately **72.5%** of actual defaults.  

This indicates strong class imbalance, with the model biased toward predicting non-default. This makes the model not indicated for real-world credit risk assessment as it is innefective predicting high-risk customers.

### Single Hidden-Layer Neural Network

A single hidden-layer neural network with dropout regularisation was also evaluated. The model consists of:

- **Input layer**: 10 hidden units with ReLU activations and dropout regularisation;
- **Output layer**: 1 unit for binary classification.

Several configurations were tested:

- **Initial setup (50 epochs, dropout = 0.4, threshold = 0.5):**  
  - Accuracy: ~96.7%;
  - Recall: ~11.5%;
  - Performed worse than logistic regression in identifying defaulters.

- **Improved setup (200 epochs, dropout = 0.1, threshold = 0.5):**  
  - Accuracy: ~97%;
  - Recall: ~27.5%;
  - Comparable to logistic regression.

- **Final setup (200 epochs, dropout = 0.1, threshold = 0.1):**  
  - Accuracy: ~95.7%;  
  - **Recall: ~47%**;  
  - Significant improvement in identifying defaulters. 

Lowering the classification threshold increased recall by making the model less conservative when predicting the minority class (`Yes`). This is particularly important in imbalanced datasets, where standard thresholds (0.5) often lead to poor detection of rare events.

---

## Conclusion

The neural network achieved the highest recall (**~47%**) after tuning the classification threshold, significantly improving its ability to detect defaulting customers compared to logistic regression (**~27.5%**).  

However, this improvement comes at the cost of reduced accuracy and increased model complexity. Logistic regression remains a strong baseline due to its simplicity, interpretability, and minimal tuning requirements.  

From the perspective of Occam’s Razor, logistic regression would generally be preferred. Nevertheless, in applications where identifying high-risk customers is critical, the improved recall of the neural network may justify its additional complexity and tuning effort.


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
