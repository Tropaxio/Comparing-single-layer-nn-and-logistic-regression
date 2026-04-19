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

### Neural Network

The neural network considered has one hidden layer and one output layer with the following architecture:

Hidden Layer:
- Apply an affine linear transformation to the incoming data, mapping 3 predictors to 10 hidden units, resulting in 40 parameters to be estimated (30 weights from the connections and 10 biases);
- Performs ReLU on the results;
- Applies dropout regularisation with probability `p=0.4`.
Output Layer:
- Apply an affine linear transformation to the incoming data where 10 predictors come in and 1 goes out, resulting in 10 weights and 1 bias. Hence, 11 parameters.

The summary of the neural network can be seen below:
```text
DefaultModel(
  (hidden_layer): Sequential(
    (0): Linear(in_features=3, out_features=10, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.4, inplace=False)
  )
  (output_layer): Linear(in_features=10, out_features=1, bias=True)
)

===================================================================================================================
Layer (type:depth-idx)                   Input Shape               Output Shape              Param #
===================================================================================================================
DefaultModel                             [8000, 3]                 [8000, 1]                 --
├─Sequential: 1-1                        [8000, 3]                 [8000, 10]                --
│    └─Linear: 2-1                       [8000, 3]                 [8000, 10]                40
│    └─ReLU: 2-2                         [8000, 10]                [8000, 10]                --
│    └─Dropout: 2-3                      [8000, 10]                [8000, 10]                --
├─Linear: 1-2                            [8000, 10]                [8000, 1]                 11
===================================================================================================================
Total params: 51
```


