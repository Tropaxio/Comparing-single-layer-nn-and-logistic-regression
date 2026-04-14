from sklearn.linear_model import LogisticRegression

def logistic_regression(
        fit_intercept,
        max_iter
):
    return LogisticRegression(
        fit_intercept=fit_intercept,
        solver='lbfgs',
        max_iter=max_iter
    )