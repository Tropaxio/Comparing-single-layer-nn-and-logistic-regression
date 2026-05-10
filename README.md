# Comparing a Single-Hidden-Layer Neural Network with Linear Logistic Regression

This project compares the performance of a single-hidden-layer neural network with dropout regularisation against linear logistic regression. The classification task is to predict whether customers will default on their credit card debt. The dataset is included in the files and contains information on 10,000 customers.

## Quick Results

| Model | Accuracy | Recall for Defaults |
|---|---:|---:|
| Logistic Regression (No resampling) | 96.95% | 27.5% |
| Logistic Regression (Random oversampling) | ~86.7% | ~86% |
| Logistic Regression (Random undersampling) | ~86.7% | ~86% |
| Logistic Regression (SMOTE) | ~88.35% | ~85.5% |
| Logistic Regression (SMOTE-Tomek) | ~88.1% | ~85.5% |
| Neural Network (No resampling) | ~96.7% | ~11.5% |
| Neural Network (Tuned) | ~97.0% | ~27.5% |
| Neural Network (Threshold 0.1) | ~95.7% | ~47.0% |
| Neural Network (Random oversampling) | ~72.8% | ~97.1% |
| Neural Network (SMOTE) | ~77.9% | ~95.6% |
| Neural Network (SMOTE-Tomek) | ~78.8% | ~92.7% |

## Project Structure

- `src` - source for project architecture, models and evaluations;
- `results` - detailed report of the results and visual plots (see [results/report.md](results/report.md)).

## The Dataset

The dataset is a simulated set containing information on ten thousand customers. The response variable is `default` (`Yes`/`No`). Other variables include:

- `student`: whether the customer is a student (`Yes`/`No`);
- `balance`: average balance remaining after the monthly payment;
- `income`: customer income.

For more information and full experiment details see [results/report.md](results/report.md).

## Methods

Data processing and modelling steps:

- The CSV dataset is loaded into a Pandas DataFrame and rows with missing values are removed (no missing values were found in this dataset).
- Predictors and the response are separated; predictors are classified as numerical or categorical.
- The data are split into training and test sets using `train_test_split` from `sklearn.model_selection`.
- Numerical features are standardised and categorical features are one-hot encoded using `ColumnTransformer` from `sklearn.compose`.
- For the neural network, data are converted from Pandas DataFrames into PyTorch tensors (cast to 32-bit floats) and the target is transformed into
a tensor with a single output column.
- Tensors are wrapped into `TensorDataset` objects and loaded into `DataLoader` for mini-batch training.

The repository `src` contains the code to reproduce preprocessing, modelling and evaluation.

---

## Results and Interpretation

### Logistic Regression

The logistic regression baseline achieved high overall accuracy (**96.95%**) but low recall for defaults (**~27.5%**), identifying only 19 of 69 defaulters in the test set. This is a consequence of strong class imbalance.

Resampling the training set improves recall at the cost of accuracy. Experiments included:

- Random oversampling / undersampling: raised recall substantially (recall ~86%) while reducing accuracy (to ~86.7% in one run).
- SMOTE oversampling: produced higher recall with better accuracy trade-offs (accuracy ~88.35%, recall ~85.5%).
- SMOTE-Tomek: similar behaviour to SMOTE (accuracy ~88.1%, recall ~85.5%).

These experiments show the typical precision–recall trade-off on imbalanced datasets: resampling increases detection of the minority class but may reduce overall accuracy.

### Single Hidden-Layer Neural Network

The neural network used has one hidden layer (10 units, ReLU) with dropout and a single output unit for binary classification. The model summary and parameter counts are given in the detailed report ([results/report.md](results/report.md)).

Key experiments and findings:

- Initial setup (50 epochs, dropout=0.4, threshold=0.5): accuracy ~96.7%, recall ~11.5%.
- Tuning (200 epochs, dropout=0.1, threshold=0.5): accuracy ~97.0%, recall ~27.5% (comparable to logistic regression).
- Lowering the classification threshold (200 epochs, dropout=0.1, threshold=0.1): accuracy ~95.7%, recall ~47.0% (best recall observed).

Resampling with the neural network:

- Random oversampling: recall up to ~97.1% but accuracy dropped to ~72.8% (many false positives).
- SMOTE oversampling: recall ~95.6% with accuracy ~77.9%.
- SMOTE-Tomek: recall ~92.7% with accuracy ~78.8%.

Overall, the neural network with SMOTE or with a lower classification threshold achieves much higher recall, at the expected cost in accuracy.

---

## Conclusion

If the objective is to maximise recall for the minority class, the best-performing setup in these experiments was the neural network trained with SMOTE. This configuration detects many more defaulters but reduces overall accuracy and increases false positives.

Logistic regression remains a strong, interpretable baseline and performs competitively after resampling. Choosing between models and thresholds depends on the cost, interpretability, scalability and maintainability as well of false negatives vs false positives.

## How to run

1) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\Activate.ps1 # Windows PowerShell
```

2) Install Dependencies
```bash 
pip install -r requirements.txt
```

3) Run the Project 
```bash
python src/main.py
```

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
