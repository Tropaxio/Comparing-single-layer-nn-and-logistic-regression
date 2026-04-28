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

Prior to training the neural network, the dataset was prepared and transformed into a format compatible with PyTorch. The predictors and target variables were converted from Pandas DataFrames into PyTorch tensors. All data were cast to `32-bit` floating point format and the target variable was reshaped into a two-dimensional tensor with a single output column to match the expected input shape of the loss function.

The tensors were then wrapped into TensorDataset objects and then transformed into DataLoader objects to facilitate iterations, mini-batch training, with a batch size of `32` and shuffling enabled.

## Methods, Models and Results 

### Logistic Regression

A logistic regression model is fitted to the training data, and its performance is then evaluated. The model achieves an accuracy of 96.95%, with the following confusion matrix:

| | Pred: No | Pred: Yes |
| :--- | :---: | :---: |
| **Actual: No** | 1,920 | 11 |
| **Actual: Yes** | 50 | 19 |

The recall score is only about **27.5%**. This shows that there is considerable class inbalance on the data, that is, there is much more `No` values than `Yes` values, making the model biased towards `No`. Since it only correctly predicted 27.5% of customers who actually defaulted, missing **72.5%** of those observations. 

If the linear logistic regression model were considered for a real credit system it would fail to flag most risky customers, leading to a significant financial loss. 

### Neural Network

The neural network considered has one hidden layer and one output layer with the following architecture:

Hidden Layer:
- Apply an affine linear transformation to the incoming data, mapping `3` predictors to `10` hidden units, resulting in `40` parameters to be estimated (`30` weights from the connections and `10` biases);
- Performs ReLU on the results;
- Applies dropout regularisation with probability `p=0.4`.
Output Layer:
- Apply an affine linear transformation to the incoming data where `10` predictors come in and `1` goes out, resulting in `10` weights and `1` bias. Hence, `11` parameters.

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
On the first run, the hyperparameters chosen are `50` epochs with batch_size of `32` and a probability threshold of 0.5. On the tests, the `loss` after `50` epochs was around `21`, having a **96.7% accuracy**, but a recall score significantly lower, when compared to logistic regression, of around 11.5%. The confusion matrix obtained is:
| | Pred: No | Pred: Yes |
| :--- | :---: | :---: |
| **Actual: No** | 1,927 | 4 |
| **Actual: Yes** | 59 | 10 |

A second experiment was conducted with `200` epochs and a reduced dropout probability of `p=0.1`. The loss remained more or less the same, now around `19`, while the **accuracy increased slightly to 97%**. The recall score to about the same of the logistic regression, **27.5%**. The confusion matrix is:
| | Pred: No | Pred: Yes |
| :--- | :---: | :---: |
| **Actual: No** | 1921 | 10 |
| **Actual: Yes** | 50 | 19 |

A third experiment maintained `200` epochs, a dropout probability of `p=0.1`, but reduced the classification threshold of predicting `yes` from `0.5` to `0.1`. Given the class inbalance towards `no`, the dataset is biased towards this value, leading to a model that is too conservative in predicting `yes`, and so by dropping the threshold a much better recall score was obtained, of around **47%**, although accuracy score dropped to **95.7%**.

## Conclusion 

Across all experiments, the neural network configuration with `200` epochs, a dropout probability of `p=0.1`, and a classification threshold of `0.1` achieved the highest recall.

However, this improvement comes at the cost of reduced accuracy, illustrating the trade-off between correctly identifying positive cases (recall) and avoiding false positives. This trade-off is particularly relevant in imbalanced datasets, where standard accuracy metrics can be misleading.

Overall, the results suggest that model performance is strongly influenced not only by the choice of model (logistic regression vs neural network), but also by the classification threshold and the underlying class distribution. In this case, logistic regression performs competitively due to the simplicity of the data, while the neural network provides additional flexibility but requires careful tuning to achieve comparable or improved results.
