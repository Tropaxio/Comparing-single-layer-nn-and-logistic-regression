## The Dataset 

The dataset consists of simulated data containing information on ten thousand customers. The objective is to predict which customers will default on their credit card debt. 
The variables in the dataset are the following:

- `default`: A factor with levels `No` and `Yes` indicating whether the customer defaulted on their debt.
- `student`:  A factor with levels `No` and `Yes` indicating whether the customer is a student.
- `balance`:  The average balance that the customer has remaining on their credit card after making their monthly payment.
- `income`: Income of customer.

## Data Processing 

The CSV dataset is converted into a Pandas DataFrame, and rows with missing values are removed. The predictors and response variable are then separated, and the predictors are classified as categorical or numerical.

The data is split into training and test sets using `train_test_split` from the `model_selection` module in scikit-learn. Numerical predictors are standardised, and categorical predictors are one-hot encoded using `ColumnTransformer` from the compose module.

## Methods, Models and Results 

### Logistic Regression

A logistic regression model is fitted to the training data, and its performance is then evaluated. The model achieves an accuracy of 96.95%, with the following confusion matrix:

| | Pred: No | Pred: Yes |
| :--- | :---: | :---: |
| **Actual: No** | 1,920 | 11 |
| **Actual: Yes** | 50 | 19 |

The recall score is only of about 27.5%. This shows that there is considerable class inbalance on the data, that is, there is much more `No` values then `Yes` values, making the model biased towards `No`. Since it only correctly predicted 27.5% of customers who actually defaulted, missing 72.5% of those observations. 

If the linear logistic regression model was to be considered for a real credit system it would fail to flag most risky customers, leading to a significant financial loss. 

